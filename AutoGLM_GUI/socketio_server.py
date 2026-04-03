"""Socket.IO server for Scrcpy video streaming."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from typing import NotRequired
from typing_extensions import TypedDict

import socketio

from AutoGLM_GUI.logger import logger
from AutoGLM_GUI.scrcpy_protocol import ScrcpyMediaStreamPacket
from AutoGLM_GUI.scrcpy_stream import ScrcpyStreamer


class VideoPacketPayload(TypedDict):
    type: str
    data: bytes
    timestamp: int
    keyframe: NotRequired[bool | None]
    pts: NotRequired[int | None]


sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    server_kwargs={"socketio_path": "/socket.io"},
)

MAX_CONCURRENT_STREAMS = 5
STREAM_START_DELAY = 1.5
STREAM_KEEPALIVE_SECONDS = 30

_stream_start_semaphore: asyncio.Semaphore | None = None
_stream_wait_queue: asyncio.Queue[tuple[str, dict[str, Any]]] | None = None
_stream_cleanup_task: asyncio.Task[None] | None = None

_socket_streamers: dict[str, ScrcpyStreamer] = {}
_stream_tasks: dict[str, asyncio.Task[None]] = {}
_device_locks: dict[str, asyncio.Lock] = {}
_device_keepalive: dict[str, float] = {}


def _get_semaphore() -> asyncio.Semaphore:
    global _stream_start_semaphore
    if _stream_start_semaphore is None:
        _stream_start_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
    return _stream_start_semaphore


def _get_wait_queue() -> asyncio.Queue[tuple[str, dict[str, Any]]]:
    global _stream_wait_queue
    if _stream_wait_queue is None:
        _stream_wait_queue = asyncio.Queue()
    return _stream_wait_queue


def _ensure_cleanup_task() -> None:
    global _stream_cleanup_task
    existing_task = _stream_cleanup_task
    if existing_task is None or existing_task.done():
        new_task = asyncio.create_task(_cleanup_stale_streams())
        _stream_cleanup_task = new_task


async def _cleanup_stale_streams() -> None:
    while True:
        await asyncio.sleep(10)
        now = time.time()
        stale_devices = [d for d, t in _device_keepalive.items() if t < now]
        for device_id in stale_devices:
            logger.info(
                f"Cleaning up stale stream for device {device_id} (keepalive expired)"
            )
            stop_streamers(device_id=device_id)
            _device_keepalive.pop(device_id, None)


async def _stop_stream_for_sid(sid: str) -> None:
    task = _stream_tasks.pop(sid, None)
    if task:
        task.cancel()

    streamer = _socket_streamers.pop(sid, None)
    if streamer:
        if streamer.device_id:
            _device_keepalive.pop(streamer.device_id, None)
        streamer.stop()


def _classify_error(exc: Exception) -> dict[str, Any]:
    """Classify error and return user-friendly message."""
    error_str = str(exc)

    if "Address already in use" in error_str or (
        "Port" in error_str and "occupied" in error_str
    ):
        return {
            "message": "端口冲突，视频流端口仍被占用。通常会自动解决，如果持续出现请重启应用。",
            "type": "port_conflict",
            "technical_details": error_str,
        }
    elif "Device" in error_str and (
        "not available" in error_str or "not found" in error_str
    ):
        return {
            "message": "设备无响应，请检查 USB/WiFi 连接。",
            "type": "device_offline",
            "technical_details": error_str,
        }
    elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
        return {
            "message": "连接超时，请检查设备连接后重试。",
            "type": "timeout",
            "technical_details": error_str,
        }
    elif "Failed to connect" in error_str:
        return {
            "message": "无法连接到 scrcpy 服务器，请检查设备连接。",
            "type": "connection_failed",
            "technical_details": error_str,
        }
    else:
        return {
            "message": error_str,
            "type": "unknown",
            "technical_details": error_str,
        }


def stop_streamers(device_id: str | None = None) -> None:
    """Stop active scrcpy streamers (all or by device)."""
    sids = list(_socket_streamers.keys())
    for sid in sids:
        streamer = _socket_streamers.get(sid)
        if not streamer:
            continue
        if device_id and streamer.device_id != device_id:
            continue
        task = _stream_tasks.pop(sid, None)
        if task:
            task.cancel()
        streamer.stop()
        _socket_streamers.pop(sid, None)


async def _stream_packets(sid: str, streamer: ScrcpyStreamer) -> None:
    try:
        async for packet in streamer.iter_packets():
            payload = _packet_to_payload(packet)
            await sio.emit("video-data", payload, to=sid)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Video streaming failed: %s", exc)
        try:
            await sio.emit("error", {"message": str(exc)}, to=sid)
        except Exception as emit_exc:
            logger.debug(
                "Failed to emit Socket.IO stream error to %s: %s", sid, emit_exc
            )
    finally:
        await _stop_stream_for_sid(sid)


def _packet_to_payload(packet: ScrcpyMediaStreamPacket) -> VideoPacketPayload:
    payload: VideoPacketPayload = {
        "type": packet.type,
        "data": packet.data,
        "timestamp": int(time.time() * 1000),
    }
    if packet.type == "data":
        payload["keyframe"] = packet.keyframe
        payload["pts"] = packet.pts
    return payload


@sio.event
async def connect(sid: str, environ: dict[str, Any]) -> None:
    logger.info("Socket.IO client connected: %s", sid)


@sio.event
async def disconnect(sid: str) -> None:
    logger.info("Socket.IO client disconnected: %s", sid)
    global _stream_cleanup_task
    if _stream_cleanup_task is None or _stream_cleanup_task.done():
        _stream_cleanup_task = asyncio.create_task(_cleanup_stale_streams())
    await _stop_stream_for_sid(sid)
    await _process_wait_queue()


@sio.on("connect-device")
async def connect_device(sid: str, data: dict[str, Any] | None) -> None:
    global _stream_cleanup_task
    if _stream_cleanup_task is None or _stream_cleanup_task.done():
        _stream_cleanup_task = asyncio.create_task(_cleanup_stale_streams())

    payload = data or {}
    device_id = payload.get("device_id") or payload.get("deviceId")
    if not device_id:
        await sio.emit(
            "error",
            {"message": "Device ID is required", "type": "invalid_request"},
            to=sid,
        )
        return

    max_size = int(payload.get("maxSize") or 1280)
    bit_rate = int(payload.get("bitRate") or 4_000_000)

    await _stop_stream_for_sid(sid)

    if device_id not in _device_locks:
        _device_locks[device_id] = asyncio.Lock()

    device_lock = _device_locks[device_id]

    semaphore = _get_semaphore()

    if semaphore.locked():
        stream_wait_queue = _get_wait_queue()
        queue_position = stream_wait_queue.qsize() + 1
        await stream_wait_queue.put((sid, payload))
        await sio.emit(
            "stream-queued",
            {
                "message": f"Too many streams starting. Your request is queued (position: {queue_position}).",
                "queue_position": queue_position,
                "max_concurrent": MAX_CONCURRENT_STREAMS,
            },
            to=sid,
        )
        logger.info(
            f"Stream request queued for {device_id} (sid: {sid}), queue position: {queue_position}"
        )
        return

    await semaphore.acquire()

    async with device_lock:
        logger.debug(f"Acquired device lock for {device_id}, sid: {sid}")

        sids_to_stop = [
            s
            for s, streamer in _socket_streamers.items()
            if s != sid and streamer.device_id == device_id
        ]
        for s in sids_to_stop:
            logger.info(f"Stopping existing stream for device {device_id} from sid {s}")
            await _stop_stream_for_sid(s)

        streamer = ScrcpyStreamer(
            device_id=device_id,
            max_size=max_size,
            bit_rate=bit_rate,
        )

        try:
            await streamer.start()
            metadata = await streamer.read_video_metadata()
            await sio.emit(
                "video-metadata",
                {
                    "deviceName": metadata.device_name,
                    "width": metadata.width,
                    "height": metadata.height,
                    "codec": metadata.codec,
                },
                to=sid,
            )

            _socket_streamers[sid] = streamer
            _stream_tasks[sid] = asyncio.create_task(_stream_packets(sid, streamer))
            _device_keepalive[device_id] = time.time() + STREAM_KEEPALIVE_SECONDS

        except Exception as exc:
            streamer.stop()
            logger.exception("Failed to start scrcpy stream: %s", exc)
            error_info = _classify_error(exc)
            await sio.emit("error", error_info, to=sid)
        finally:
            semaphore.release()
            await _process_wait_queue()


async def _process_wait_queue() -> None:
    stream_wait_queue = _get_wait_queue()
    if stream_wait_queue.empty():
        return

    await asyncio.sleep(STREAM_START_DELAY)

    if stream_wait_queue.empty():
        return

    next_sid, next_payload = await stream_wait_queue.get()
    next_device_id = next_payload.get("device_id") or next_payload.get("deviceId")

    if not next_device_id:
        logger.warning(f"Cannot process queued stream: missing device_id")
        return

    semaphore = _get_semaphore()
    if semaphore.locked():
        await stream_wait_queue.put((next_sid, next_payload))
        return

    await semaphore.acquire()

    asyncio.create_task(
        _start_queued_stream(next_sid, next_device_id, next_payload, semaphore)
    )


async def _start_queued_stream(
    sid: str, device_id: str, payload: dict[str, Any], semaphore: asyncio.Semaphore
) -> None:
    max_size = int(payload.get("maxSize") or 1280)
    bit_rate = int(payload.get("bitRate") or 4_000_000)

    if device_id not in _device_locks:
        _device_locks[device_id] = asyncio.Lock()

    device_lock = _device_locks[device_id]

    async with device_lock:
        await _stop_stream_for_sid(sid)

        logger.info(f"Starting queued stream for device {device_id}, sid: {sid}")

        streamer = ScrcpyStreamer(
            device_id=device_id,
            max_size=max_size,
            bit_rate=bit_rate,
        )

        try:
            await streamer.start()
            metadata = await streamer.read_video_metadata()
            await sio.emit(
                "stream-ready",
                {
                    "deviceName": metadata.device_name,
                    "width": metadata.width,
                    "height": metadata.height,
                    "codec": metadata.codec,
                    "message": "Your queued stream is now ready",
                },
                to=sid,
            )

            _socket_streamers[sid] = streamer
            _stream_tasks[sid] = asyncio.create_task(_stream_packets(sid, streamer))
            _device_keepalive[device_id] = time.time() + STREAM_KEEPALIVE_SECONDS

        except Exception as exc:
            streamer.stop()
            logger.exception("Failed to start queued scrcpy stream: %s", exc)
            error_info = _classify_error(exc)
            await sio.emit("error", error_info, to=sid)
        finally:
            semaphore.release()
            await _process_wait_queue()

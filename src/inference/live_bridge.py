"""
Live Python<->DLL bridge for world-model inference.

Transport: a TCP loopback socket. Python (running in WSL2) is the LISTENER;
the DLL (running in Windows) DIALS OUT to 127.0.0.1:WM_PORT. This relies on
WSL2's default NAT networking mode, which auto-forwards Windows-side
connections to 127.0.0.1:<port> into whatever is listening inside the VM —
no manual IP discovery needed. (Named pipes can't be used here at all:
they're a Windows-kernel-only object and don't exist across the WSL2 VM
boundary.)

Implements bridge_protocol.GameBridge exactly:
  - reset_to_start() and step() are both single synchronous round trips
    from Python's point of view, no matter how many raw physics ticks
    a reset costs internally on the DLL side.
  - step()'s hold_ticks is handled ENTIRELY here: we just send `hold_ticks`
    consecutive TAG_STEP messages with the same action and keep only the
    telemetry from the last one — mirroring windowing.read_raw_window's
    stride sampling exactly, same as the docstring requires. The DLL has
    no concept of action-hold at all; it only ever sees 1 tick per message.
"""
from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from action_codec import GameAction

from ingest_raw import _POS_COLS, _QUAT_COLS, _VEL_COLS, _AVel_COLS, _DAMP_COLS, _WHEL_COLS

RAW_VECTOR_GROUPS: dict[str, tuple[str, ...]] = {
    'pos':           _POS_COLS,
    'quat':          _QUAT_COLS,
    'vel_world':     _VEL_COLS,
    'angvel_world':  _AVel_COLS,
    'damper_len':    _DAMP_COLS,
    'wheel_rot_spd': _WHEL_COLS,
}

# scalar-valued HDF5_RAW_KEYS -> single source field name
RAW_SCALAR_FIELDS: dict[str, str] = {
    'cur_gear':    'CurGear',
    'input_steer': 'InputSteer',
    'input_gas':   'InputForward',
    'input_brake': 'InputBackward/brake',
}

# not part of HDF5_RAW_KEYS, but required by bridge_protocol.GameBridge.step()
RESPAWN_FLAG_FIELDS: dict[str, str] = {
    'launched_respawn': 'LaunchedRespawn',
    'static_respawn':   'StaticRespawn',
}

REQUIRED_FIELD_NAMES: frozenset[str] = (
    frozenset(n for cols in RAW_VECTOR_GROUPS.values() for n in cols)
    | frozenset(RAW_SCALAR_FIELDS.values())
    | frozenset(RESPAWN_FLAG_FIELDS.values())
)

# ── IPC tags (must match the DLL) ────────────────────────────────────────
TAG_SCHEMA   = 1
TAG_OBS      = 2
TAG_STEP     = 10
TAG_RESET    = 11
TAG_SHUTDOWN = 12

# ── Transport config ──────────────────────────────────────────────────────
# Bind wide (0.0.0.0) inside the WSL2 VM so the Windows-side forwarder can
# actually reach it — binding 127.0.0.1 only works for connections that
# already originate inside the VM. Must match WM_PORT on the DLL side.
WM_BIND_HOST = "0.0.0.0"
WM_PORT      = 47821

SINPUT_STRUCT_FMT = "<IffffhhI"
assert struct.calcsize(SINPUT_STRUCT_FMT) == 28

# FieldType (field_types.h) -> struct format char. Sizes must match
# TMShared::FieldSize() exactly; signedness/float-ness matters here even
# though FieldSize() only cares about byte width.
FIELD_TYPE_FMT = {
    0:  'b',  # T_I8
    1:  'B',  # T_U8
    2:  'B',  # T_UNK8
    3:  'h',  # T_I16
    4:  'H',  # T_U16
    5:  'H',  # T_UNK16
    6:  'i',  # T_I32
    7:  'I',  # T_U32
    8:  'f',  # T_F32
    9:  'I',  # T_UNK32
    10: 'q',  # T_I64
    11: 'Q',  # T_U64
    12: 'd',  # T_F64
    13: 'Q',  # T_UNK64
    14: 'Q',  # T_PTR
    15: '?',  # T_BOOL
}

_WIRE_FIELD_DEF_FMT = "<64sIB"
_WIRE_FIELD_DEF_SIZE = struct.calcsize(_WIRE_FIELD_DEF_FMT)


@dataclass
class FieldSchema:
    names: list[str]
    types: list[int]
    parser: struct.Struct  # decodes one car's raw field blob -> tuple


def _accept_server(port: int, bind_host: str = WM_BIND_HOST, timeout: float = 60.0) -> socket.socket:
    """Listen for and accept exactly one incoming connection (from the DLL),
    then hand back a connected, TCP_NODELAY'd socket."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_host, port))
    srv.listen(1)
    srv.settimeout(timeout)
    print("Waiting for DLL to connect...")
    try:
        conn, addr = srv.accept()
    finally:
        srv.close()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[bridge] DLL connected from {addr}")
    return conn


class SocketFile:
    """Minimal file-like adapter over a connected socket, so the rest of the
    bridge (self.r.read / self.w.write / self.w.flush) needs zero changes
    from the named-pipe version."""

    def __init__(self, sock: socket.socket):
        self.sock = sock

    def read(self, n: int) -> bytes:
        # May return < n bytes; callers already loop via _recv_exact.
        return self.sock.recv(n)

    def write(self, data: bytes) -> None:
        self.sock.sendall(data)

    def flush(self) -> None:
        pass  # TCP_NODELAY means nothing is buffered on our side anyway

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


def pack_single_input(action: GameAction) -> bytes:
    """
    GameAction -> one raw 28-byte SInputState.

    World-model actions only cover steer/gas/brake — statesWord stays at
    plain EInputMode::Analog (2), buttonsWord/axis3/mouse stay neutral.
    (Contrast with the replay bridge's pack_inputs(), which has to
    reproduce every button/state bit a recorded ghost used.)
    """
    steer = max(-1.0, min(1.0, float(action.steer)))
    gas   = 1.0 if action.gas else 0.0
    brake = 1.0 if action.brake else 0.0
    return struct.pack(SINPUT_STRUCT_FMT, 2, steer, gas, brake, 0.0, 0, 0, 0)


class TMWorldModelBridge:
    """Implements bridge_protocol.GameBridge."""

    def __init__(self):
        conn = _accept_server(WM_PORT)
        # Full-duplex socket now — r and w just alias the same connection
        # (kept as two attributes so the rest of the class is unchanged).
        self.r = SocketFile(conn)
        self.w = self.r
        self.schema = self._recv_schema()
        # Hemisphere-fix continuity state (see _fix_quat_hemisphere below).
        self._last_quat_sign_ref: Optional[np.ndarray] = None

    # ── low-level framing ────────────────────────────────────────────
    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.r.read(n - len(buf))
            if not chunk:
                raise ConnectionError("DLL socket closed")
            buf.extend(chunk)
        return bytes(buf)

    def _recv(self):
        tag = struct.unpack("<I", self._recv_exact(4))[0]
        ln = struct.unpack("<I", self._recv_exact(4))[0]
        payload = self._recv_exact(ln) if ln else b""
        return tag, payload

    def _send(self, tag: int, payload: bytes = b""):
        self.w.write(struct.pack("<II", tag, len(payload)))
        if payload:
            self.w.write(payload)
        self.w.flush()

    # ── schema ────────────────────────────────────────────────────────
    def _recv_schema(self) -> FieldSchema:
        tag, payload = self._recv()
        if tag != TAG_SCHEMA:
            raise ConnectionError(f"expected TAG_SCHEMA first, got tag={tag}")

        field_count, stride = struct.unpack_from("<II", payload, 0)
        names, types, fmts = [], [], []
        off = 8
        for _ in range(field_count):
            name_raw, _foffset, ftype = struct.unpack_from(_WIRE_FIELD_DEF_FMT, payload, off)
            off += _WIRE_FIELD_DEF_SIZE
            name = name_raw.split(b"\0", 1)[0].decode("utf-8")
            if ftype not in FIELD_TYPE_FMT:
                raise ValueError(f"unknown FieldType {ftype} for field '{name}'")
            names.append(name)
            types.append(ftype)
            fmts.append(FIELD_TYPE_FMT[ftype])

        parser = struct.Struct("<" + "".join(fmts))
        if parser.size != stride:
            raise ValueError(
                f"schema stride mismatch: DLL says {stride} bytes, "
                f"decoded field list implies {parser.size} bytes"
            )

        self.missing = REQUIRED_FIELD_NAMES - set(names)
        # if missing:
        #     raise ValueError(
        #         f"fields.json is missing fields required by the world-model "
        #         f"bridge: {sorted(missing)}. Add them to fields.json (same "
        #         f"CSV column names ingest_raw.py expects) and restart the DLL."
        #     )

        print(f"[bridge] schema: {field_count} fields, stride={stride} bytes")
        return FieldSchema(names=names, types=types, parser=parser)

    # ── telemetry decode ────────────────────────────────────────────
    def _fix_quat_hemisphere(self, quat_raw: np.ndarray, is_new_segment: bool) -> np.ndarray:
        """
        Incremental version of ingest_raw.py/quaternion_utils'
        hemisphere-fix + normalize: offline it's done as one vectorized
        pass over a whole segment (comparing each frame to the previous
        one); online we only ever see one frame at a time, so we keep the
        last emitted (sign-fixed) quaternion as the reference instead.

        A fresh segment (right after reset_to_start()/a respawn) has no
        valid previous reference — canonicalize via W >= 0, matching how
        offline segments each start a fresh hemisphere-fix pass.
        """
        if is_new_segment or self._last_quat_sign_ref is None:
            if quat_raw[0] < 0.0:
                quat_raw = -quat_raw
        else:
            if np.dot(quat_raw, self._last_quat_sign_ref) < 0.0:
                quat_raw = -quat_raw
        self._last_quat_sign_ref = quat_raw.copy()
        n = np.linalg.norm(quat_raw)
        if n > 1e-8:
            quat_raw = quat_raw / n
        return quat_raw.astype(np.float32)

    def _build_telemetry(self, decoded: dict, is_new_segment: bool) -> dict:
        telem: dict = {}
        for hkey, cols in RAW_VECTOR_GROUPS.items():
            vec = np.array([float(decoded[c]) for c in cols], dtype=np.float32)
            if hkey == 'quat':
                vec = self._fix_quat_hemisphere(vec.astype(np.float64), is_new_segment)
            telem[hkey] = vec
        for hkey, col in RAW_SCALAR_FIELDS.items():
            telem[hkey] = float(decoded[col])
        for hkey, col in RESPAWN_FLAG_FIELDS.items():
            telem[hkey] = bool(decoded[col])
        return telem

    def _decode_obs(self, payload: bytes):
        tick, flags = struct.unpack_from("<IB", payload, 0)
        is_reset_result = bool(flags & 0x1)
        blob = payload[5:5 + self.schema.parser.size]
        values = self.schema.parser.unpack(blob)
        decoded = dict(zip(self.schema.names, values))
        for missing_field in self.missing: #! BAND AID FIX JUST TO TEST (but matches training)
            decoded[missing_field] = 0
        telem = self._build_telemetry(decoded, is_new_segment=is_reset_result)
        telem['tick'] = tick
        return telem, is_reset_result

    # ── GameBridge protocol ──────────────────────────────────────────
    def step(self, action: GameAction, hold_ticks: int) -> dict:
        assert hold_ticks >= 1
        payload_in = pack_single_input(action)
        telem = None
        for _ in range(hold_ticks):
            self._send(TAG_STEP, payload_in)
            tag, payload = self._recv()
            if tag != TAG_OBS:
                raise ConnectionError(f"expected TAG_OBS, got tag={tag}")
            telem, is_reset_result = self._decode_obs(payload)
            if is_reset_result:
                raise ConnectionError("got a reset-result OBS in response to TAG_STEP — protocol desync")
        return telem

    def reset_to_start(self) -> dict:
        self._send(TAG_RESET)
        tag, payload = self._recv()
        if tag != TAG_OBS:
            raise ConnectionError(f"expected TAG_OBS after TAG_RESET, got tag={tag}")
        telem, is_reset_result = self._decode_obs(payload)
        if not is_reset_result:
            raise ConnectionError("expected a reset-result OBS, got a normal step OBS — protocol desync")
        return telem

    def close(self):
        try:
            self._send(TAG_SHUTDOWN)
        except Exception:
            pass
        try:
            self.r.close()  # closes the one underlying socket for both r and w
        except Exception:
            pass
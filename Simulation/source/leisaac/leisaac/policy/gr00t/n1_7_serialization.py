"""Serializer compatible with NVIDIA Isaac-GR00T N1.7 policy server."""

import msgpack
import numpy as np

try:
    import msgpack_numpy
except ImportError:
    msgpack_numpy = None


def _encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("O", "V"):
            raise ValueError(f"Unsupported numpy dtype: {obj.dtype}")
        return {
            b"nd": True,
            b"type": obj.dtype.str,
            b"kind": b"",
            b"shape": obj.shape,
            b"data": obj.tobytes(),
        }

    if isinstance(obj, np.generic):
        return {
            b"nd": False,
            b"type": obj.dtype.str,
            b"data": obj.tobytes(),
        }

    return obj


def _decode_numpy(obj):
    if b"nd" not in obj:
        return obj

    dtype = np.dtype(obj[b"type"])
    if obj[b"nd"] is True:
        return np.ndarray(buffer=obj[b"data"], dtype=dtype, shape=obj[b"shape"])
    return np.frombuffer(obj[b"data"], dtype=dtype)[0]


class MsgSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes):
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes, raw=False)

    @staticmethod
    def decode_custom_classes(obj):
        if msgpack_numpy is not None:
            return msgpack_numpy.decode(obj)
        return _decode_numpy(obj)

    @staticmethod
    def encode_custom_classes(obj):
        if msgpack_numpy is not None:
            return msgpack_numpy.encode(obj)
        return _encode_numpy(obj)

# common/utils.py
import binascii

def hexdump(data: bytes, length=16):
    """
    Useful for debugging binary data sent over the network.
    """
    lines = []
    for i in range(0, len(data), length):
        chunk = data[i:i+length]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        text_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f'{i:04x}   {hex_part:<{length*3}}   |{text_part}|')
    return '\n'.join(lines)
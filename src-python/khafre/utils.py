from pynput.keyboard import Key, Listener
import sys
import termios

keyESCAPE = Key.esc

# Auxiliary functions to exit on key press (or rather, release)

def _on_press(key):
    pass
def _on_release(key, keyRef):
    if key == keyRef:
        # Stop listener
        return False

def repeatUntilKey(fn, key=keyESCAPE):
    with Listener(on_press=_on_press, on_release=lambda k: _on_release(k, key)) as listener:
        while True:
            ans = fn()
            if not ans:
                listener.stop()
            if not listener.running:
                break
        listener.join()
        termios.tcflush(sys.stdin, termios.TCIOFLUSH)


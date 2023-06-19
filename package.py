from dataclasses import dataclass
import pickle
from sys import getsizeof



@dataclass
class Package:
    action: str
    payload: bytes



def create_byte_package(action, payload) -> tuple[int, Package]:
    python_package = Package(action, payload)
    byte_package = pickle.dumps(python_package)

    return len(byte_package), byte_package


def load_package(byte_package):
    return pickle.loads(byte_package)



if __name__ == "__main__":
    size, byte_c = create_byte_package("start digga", 69410)
    print(size, type(size), type(byte_c), byte_c)
    new_package = load_package(byte_c)
    print(new_package.action, new_package.payload)

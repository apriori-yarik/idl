from dl_lib.perceptron import datasets, functional

def main() -> int:
    ds = datasets.create_dataset(4)
    w1 = functional.initialize_weights(0, 100, 1)[0]
    w2 = functional.initialize_weights(0, 10, 1)[0]

    print(f"Dataset: {ds}\n")
    print(f"initialize_weights(0, 100, 1): {w1}\n")
    print(f"initialize_weights(0, 10, 1): {w2}\n")

    return 0
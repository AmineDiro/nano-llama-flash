import torch


def main():
    print(
        f"cuda available: {torch.cuda.is_available()}\nbf16 support : {torch.cuda.is_bf16_supported()}"
    )


if __name__ == "__main__":
    main()

class MunicipalCodeControlDigit:
    """
    Class to calculate the control digit for a given test value.

    Translated from Java to Python from the following Github Gist:
    https://gist.github.com/eltabo/e9fa5fc1e5dd8c2140b4
    """

    magic = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 3, 8, 2, 7, 4, 1, 5, 9, 6],
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
    ]

    @staticmethod
    def calculate(municipal_code: int) -> int:
        """
        Calculates the control digit for a given test value.

        Args:
            test (int): The test value.

        Returns:
            int: The control digit.

        """
        bytes_str = format(municipal_code, "05").encode()

        total_sum = 0
        for i, v in enumerate(bytes_str):
            total_sum += MunicipalCodeControlDigit.magic[2 - i % 3][v - 48]

        return 0 if total_sum == 0 else 10 - total_sum % 10

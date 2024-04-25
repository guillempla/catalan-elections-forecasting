class MunicipalCodeControlDigit:
    """
    Class to calculate the control digit for a given municipal code.
    """

    magic = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 3, 8, 2, 7, 4, 1, 5, 9, 6],
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
    ]

    @staticmethod
    def calculate(municipal_code: int) -> int:
        """
        Calculates the control digit for a given municipal code.

        Args:
            municipal_code (int): The municipal code.

        Returns:
            int: The control digit.
        """
        # Ensure the municipal code is at least 5 digits, pad with zeros if necessary
        bytes_str = format(municipal_code, "05").encode()

        total_sum = 0
        for i, v in enumerate(bytes_str):
            total_sum += MunicipalCodeControlDigit.magic[2 - i % 3][v - 48]

        # Return 0 directly if total_sum modulo 10 is 0, otherwise calculate the difference to 10
        result = 10 - total_sum % 10
        return 0 if result == 10 else result


if __name__ == "__main__":
    # Example usage
    municipal_code = 43123
    control_digit = MunicipalCodeControlDigit.calculate(municipal_code)
    print(f"The control digit for municipal code {municipal_code} is {control_digit}")

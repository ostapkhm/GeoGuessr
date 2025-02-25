class DistrictParser:
    """Handles robust parsing of user input for districts."""

    @staticmethod
    def get_int_input(prompt, min_val=1, max_val=None):
        # Get an integer input from the user with validation.

        while True:
            try:
                value = int(input(prompt).strip())
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}. Try again.")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must not exceed {max_val}. Try again.")
                    continue
                
                return value
            except ValueError:
                print("Invalid input! Please enter a valid integer.")

    @staticmethod
    def get_float_input(prompt):
        # Get a floating-point input from the user with validation.

        while True:
            try:
                return float(input(prompt).strip())
            except ValueError:
                print("Invalid input! Please enter a valid number.")

    @staticmethod
    def get_districts():
        # Prompt the user to input district parameters safely.

        n_districts = DistrictParser.get_int_input("Specify the number of districts for image retrieval: ", 1)

        centeres = []
        r_mins = []
        lengths = []
        n_points = []

        for i in range(n_districts):
            print(f"\nDistrict {i+1}:")
            lat = DistrictParser.get_float_input("Enter latitude: ")
            lon = DistrictParser.get_float_input("Enter longitude: ")
            r_min = DistrictParser.get_int_input("Enter radius of neighborhood sample (meters): ", 1)
            length = DistrictParser.get_int_input("Enter length of district square (meters): ", 1)
            points_nb = DistrictParser.get_int_input("Enter max number of points to sample in the district: ", 1)

            centeres.append([lat, lon])
            r_mins.append(r_min)
            lengths.append(length)
            n_points.append(points_nb)

        return centeres, r_mins, lengths, n_points
import csv


class LinearRegression:
    def __init__(self, file_path):
        self.file_path = file_path

        self.years_experience = []
        self.salaries = []

        self.mean_years_experience = 0.0
        self.mean_salary = 0.0
        self.median_years_experience = 0.0
        self.median_salary = 0.0

        self.slope = 0.0  # beta_1
        self.intercept = 0.0  # beta_0

        self.read_data()
        self.compute_basic_statistics()
        self.compute_regression_parameters()

    def read_data(self):
        with open(self.file_path, newline="") as csvfile:
            reader = csv.reader(csvfile)

            # Skip the header row
            next(reader, None)

            for row in reader:
                if len(row) < 2:
                    continue

                x = float(row[0])
                y = float(row[1])

                self.years_experience.append(x)
                self.salaries.append(y)

    def compute_basic_statistics(self):
        n = len(self.salaries)
        if n == 0:
            raise ValueError("No data points found in the CSV file.")

        self.years_experience.sort()
        self.salaries.sort()

        total_x = sum(self.years_experience)
        total_y = sum(self.salaries)
        self.mean_years_experience = total_x / n
        self.mean_salary = total_y / n

        if n % 2 == 1:
            self.median_years_experience = self.years_experience[n // 2]
            self.median_salary = self.salaries[n // 2]
        else:
            mid1 = n // 2
            mid2 = mid1 - 1
            self.median_years_experience = (
                self.years_experience[mid1] + self.years_experience[mid2]
            ) / 2
            self.median_salary = (self.salaries[mid1] + self.salaries[mid2]) / 2

    def compute_regression_parameters(self):
        n = len(self.salaries)

        x_mean = self.mean_years_experience
        y_mean = self.mean_salary

        numerator = 0.0
        denominator = 0.0

        for x, y in zip(self.years_experience, self.salaries):
            numerator += (x - x_mean) * (y - y_mean)
            denominator += (x - x_mean) ** 2

        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * x_mean)

    def predict(self, x):
        return self.intercept + self.slope * x


if __name__ == "__main__":
    lr_model = LinearRegression("salary_data.csv")

    print(f"Mean Years of Experience: {lr_model.mean_years_experience}")
    print(f"Mean Salary: {lr_model.mean_salary}")
    print(f"Median Years of Experience: {lr_model.median_years_experience}")
    print(f"Median Salary: {lr_model.median_salary}")
    print(f"Slope (beta_1): {lr_model.slope}")
    print(f"Intercept (beta_0): {lr_model.intercept}")

    years_experience_test = 5.0
    predicted_salary = lr_model.predict(years_experience_test)
    print(
        f"Predicted salary for {years_experience_test} years of experience: {predicted_salary:.2f}"
    )

import pandas as pd


def load_text_file(filename):
    with open(f"./resources/{filename}.txt", "r") as file:
        content = file.read()
        return content.splitlines()


if __name__ == "__main__":
    # Download CSV here: https://www.kaggle.com/datasets/peopledatalabssf/free-7-million-company-dataset?resource=download
    df = pd.read_csv("./resources/companies_sorted.csv.zip", compression="zip")
    all_names = "\n".join(df["name"].astype(str))

    with open("./resources/company_names.txt", "w", encoding="utf-8") as f:
        f.write(all_names)

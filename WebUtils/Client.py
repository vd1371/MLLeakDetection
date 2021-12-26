import pandas as pd
import requests
import io

def make_request(batch_number = 123):

    x = requests.get(f"http://127.0.0.1/?batch_number={batch_number}").content.decode("utf-8")
    print('--------------------------------------------------',x,'--------------------------------------------------------------------------------')

    if not "NotFound" in x:
        df = pd.read_csv(io.StringIO(x), header = 0, index_col = 0)
    print (df)


if __name__ == "__main__":

    make_request(123)

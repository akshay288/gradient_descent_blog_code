import argparse
from tqdm import trange
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--test', required=True)
parser.add_argument('--train', required=True)
args = parser.parse_args()

def load_data(path):
    df = pd.read_csv(path)
    
    houses = []

    for _, row in df.iterrows():
        square_feet = row['GrLivArea']
        beds = row['BedroomAbvGr']
        full_baths = row['BsmtFullBath'] + row['FullBath']
        half_baths = row['BsmtHalfBath'] + row['HalfBath']

        if any(np.isnan([square_feet, beds, full_baths, half_baths])):
            continue

        houses.append(
            {
                'beds': beds,
                'baths': full_baths + (half_baths / 2),
                'squarefeet': square_feet
            }
        )

    return houses

def pred_squarefeet(beds, baths, a):
    return a * (beds + baths)

def loss(actual, expected):
    return (expected - actual) ** 2

def a_loss_partial_deriv(bed, bath, a, actual_sf):
    return ((2 * a * (bed ** 2))
            + (4 * a * bed * bath)
            + (2 * a * (bath ** 2))
            - (2 * bed * actual_sf)
            - (2 * bath * actual_sf))

LEARNING_RATE = 0.01
EPOCHS = 1000

def get_optimal_params(train_data):
    a = 0.1

    t = trange(0, EPOCHS) 
    for _ in t:
        t.set_postfix(a='{0:.2f}'.format(a))
        for datum in train_data:
            beds = datum['beds']
            baths = datum['baths']
            actual_sf = datum['squarefeet']

            a_part_deriv = a_loss_partial_deriv(beds, baths, a, actual_sf)

            if a_part_deriv < 0:
                a = a + LEARNING_RATE
            elif a_part_deriv > 0:
                a = a - LEARNING_RATE
    
    return a

train_data = load_data(args.train)
test_data = load_data(args.test)

a = get_optimal_params(train_data)
all_loss = []
errors = []
for datum in test_data:
    beds = datum['beds']
    baths = datum['baths']
    expected_squarefeet = datum['squarefeet']
    actual_squarefeet = pred_squarefeet(beds, baths, a)

    all_loss.append(loss(actual_squarefeet, expected_squarefeet))
    errors.append(abs(expected_squarefeet - actual_squarefeet) / actual_squarefeet)

avg_loss = sum(all_loss) / len(all_loss)
avg_error = sum(errors) / len(errors)

print('Learning Rate: {}'.format(LEARNING_RATE))
print('Epochs: {}'.format(EPOCHS))
print('Loss: {}'.format(avg_loss))
print('Error: {}'.format(avg_error))

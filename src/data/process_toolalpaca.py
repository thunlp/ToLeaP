import json

def process_train(filename):
    raw_train_path = filename
    train_json = json.load(open(raw_train_path, 'r'))
    train_factory = []
    # Convert to sharegpt
    for d in train_json:
        index = 0
        item = {"conversation": []}
        for message in d[0]:
            if index % 2 == 0:
                item["conversation"].append({"from": "human", "value": message})
            else:
                item["conversation"].append({"from": "gpt", "value": message})
            index += 1
        train_factory.append(item)
    json.dump(train_factory, open(filename, 'w'), indent=4)
    return train_factory

if __name__ == "__main__":
    process_train('toolalpaca_train.json')
    process_train('toolalpaca_val_sim.json')
    process_train('toolalpaca_val_real.json')
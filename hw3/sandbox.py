import json
import numpy as np
import lang_to_sem_loader

# with open("lang_to_sem_data.json") as jsf:
#     raw_data = json.load(jsf)

# data = raw_data["train"] + raw_data["valid_seen"]

# train_data = [instance for sublist in data for instance in sublist]
# lens = [len(list(x[0].split(" "))) for x in train_data]
# print(max(lens))
# train_data = np.array(train_data)
# print(train_data.shape)

# train_loader, valid_loader = lang_to_sem_loader.get_loaders()

# print(iter(train_loader).next())

lang_to_sem_loader.LangToSemDataset(input_path="lang_to_sem_data.json", is_train=True, vocab_size=1000, debug=False)

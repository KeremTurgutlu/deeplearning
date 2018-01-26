from imports import *

### CREATE DATASET CLASSES
class logSpecData(Dataset):
    def __init__(self, file_path, classes, sample_ratio=1.0):
        self.file_path = file_path
        self.classes = classes
        img_paths = []
        for c in classes:
            for img_path in os.listdir(file_path + c):
                img_paths.append(file_path + c + '/' + img_path)
        # shuffle with sorted
        # subsample for easier experimentation
        random.shuffle(img_paths)
        self.img_paths = img_paths[:int(len(img_paths) * sample_ratio)]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = img_path.split('/')[4]
        # read wav file
        sample_rate, samples = wavfile.read(img_path)
        # get mel spectogram and log mel spectogram
        S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
        log_S = librosa.power_to_db(S, ref=np.max)
        # get label index from classes
        label_idx = np.where(np.array(self.classes) == label)[0][0]
        img = log_S.T
        # resize images that are not in right size
        if img.shape != (32, 128):
            img = resize(log_S.T, (32, 128))
        # add dimension for 1 channel image
        img = img[None, :]
        return img, label_idx

    def __len__(self):
        return len(self.img_paths)


### SAMPLER
def get_sampler_weights(dataset):
    labels = [i.split('/')[4] for i in dataset.img_paths]
    count_dict = Counter(labels)
    total_count = sum(count_dict.values())
    weight_dict = {k:total_count / count_dict[k] for k in count_dict}
    weights = np.vectorize(weight_dict.get)(np.array(labels))
    return weights, weight_dict
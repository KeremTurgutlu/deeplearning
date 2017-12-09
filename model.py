
# model: A PyTorch model
# data: Model data which has cv_dls instance
# epochs: Number of epochs to train the model
# opt: Optimizer of choice
# crit: Loss criterion
# metrics: Additional metrics to calculate


def fit_cv(model, data, epochs, opt, crit, metrics=None):
    for epoch in range(epochs):
        cv_trn_losses = []
        cv_val_losses = []

        for cv_pair in tqdm(data.cv_dls):
            trn_dl = cv_pair[0]
            val_dl = cv_pair[1]

            # Tranining
            alpha = 0.98
            smooth_loss, batch_num = 0., 0
            for (x, y) in iter(trn_dl):
                batch_num += 1
                #Calculate loss
                outputs, labels = model(V(x)), V(y)
                loss = criterion(outputs, labels.squeeze(-1))
                smooth_loss = (alpha)*smooth_loss + (1 - alpha)*loss.data

                #Update Weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # debias loss
            smooth_loss = smooth_loss / (1 - alpha**batch_num)

            # Validation
            val_loss = 0
            num_obs = 0
            for (x, y) in iter(val_dl):
                outputs, labels = model(V(x)), V(y)
                loss = criterion(outputs, labels.squeeze(-1))
                n = len(labels)
                num_obs += n
                val_loss += loss.data[0]*n

            # Collect Aggreagated losses
            cv_trn_loss = smooth_loss[0]
            cv_val_loss = val_loss/num_obs
            cv_trn_losses.append(cv_trn_loss)
            cv_val_losses.append(cv_val_loss)


        cv_trn_avg = round(np.mean(cv_trn_losses), 4)
        cv_trn_std = round(np.std(cv_trn_losses), 4)
        cv_val_avg = round(np.mean(cv_val_losses), 4)
        cv_val_std = round(np.std(cv_val_losses), 4)

        print(f'epoch: {epoch}')
        print(f'cv train: {cv_trn_avg} +/- {cv_trn_std} cv val: {cv_val_avg} +/- {cv_val_std}')
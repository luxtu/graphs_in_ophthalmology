import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.optim.lr_scheduler import ExponentialLR


def smoothed_label_loss(out, y, num_classes, loss_func, device):
    """
    Apply Gaussian filter to the one-hot encoded target labels to smooth them.
    """

    smooth_y = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
    # Convert PyTorch tensor to NumPy array
    smooth_y_np = smooth_y.cpu().numpy()
    # convert to float
    smooth_y_np = smooth_y_np.astype(float)
    # Apply Gaussian filter for smoothin using scipy
    smooth_y_filtered = gaussian_filter(smooth_y_np, sigma=0.4)
    # Convert back to PyTorch tensor
    smooth_y_filtered = torch.from_numpy(smooth_y_filtered).to(device)
    # make sure its a float
    smooth_y_filtered = smooth_y_filtered.float()
    loss = loss_func(out, smooth_y_filtered)

    return loss


class graphClassifierSimple:
    def __init__(
        self, model, loss_func, lr=0.005, weight_decay=5e-5, smooth_label_loss=False
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.lossFunc = loss_func
        self.smooth_label_loss = smooth_label_loss

    def train(self, loader, data_loss_dict=None):
        self.model.train()
        cum_loss = 0
        raw_out = []
        y_out = []
        if data_loss_dict is None:
            data_loss_dict = {}

        size_data_set = len(
            loader.dataset
        )  # must be done before iterating/regenerating the dataset
        for data in loader:
            # data.to(self.device)

            out = self.model(data.x, data.edge_index, data.batch)

            # get the number of classes from the output
            num_classes = out.shape[1]
            if self.smooth_label_loss:
                loss = smoothed_label_loss(
                    out, data.y, num_classes, self.lossFunc, self.device
                )
            else:
                loss = self.lossFunc(out, data.y)

            loss.backward()
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            cum_loss += loss.item()

            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())

        # raw_out)
        pred = np.concatenate(raw_out, axis=0)
        y = np.concatenate(y_out, axis=0)
        self.scheduler.step()
        return cum_loss / size_data_set, pred, y

    @torch.no_grad()
    def predict(self, loader):
        raw_out = []
        y_out = []
        self.model.eval()
        for data in loader:
            data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())

        pred = np.concatenate(raw_out, axis=0)
        y = np.concatenate(y_out, axis=0)

        return pred, y



class graphRegressorSimple:
    def __init__(
        self, model, loss_func, lr=0.005, weight_decay=5e-5
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.95)
        self.lossFunc = loss_func

    def train(self, loader):
        self.model.train()
        cum_loss = 0
        raw_out = []
        y_out = []

        size_data_set = len(
            loader.dataset
        )  # must be done before iterating/regenerating the dataset
        for data in loader:
            # data.to(self.device)

            out = self.model(data.x, data.edge_index, data.batch)
            # remove the last dimension
            out = out.squeeze()

            # get the number of classes from the output
            loss = self.lossFunc(out, data.y)

            loss.backward()
            self.optimizer.step()  # Update parameters based on gradients.
            self.optimizer.zero_grad()  # Clear gradients.
            cum_loss += loss.item()

            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())

        # raw_out)
        pred = np.concatenate(raw_out, axis=0)
        y = np.concatenate(y_out, axis=0)
        self.scheduler.step()
        return cum_loss / size_data_set, pred, y

    @torch.no_grad()
    def predict(self, loader):
        raw_out = []
        y_out = []
        self.model.eval()
        for data in loader:
            data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            raw_out.append(out.cpu().detach().numpy())
            y_out.append(data.y.cpu().detach().numpy())

        pred = np.concatenate(raw_out, axis=0)
        y = np.concatenate(y_out, axis=0)

        return pred, y
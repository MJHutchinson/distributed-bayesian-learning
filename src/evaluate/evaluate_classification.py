import torch
from torch.utils.data.dataloader import DataLoader

def calibration_bin(model, dataset, num_bins=None, bin_edges=None):
    """ For a given model, compute the bin values for a calibration plot.
    Either takes prespecified bins, or takes a number of bins to split the data into.

    Must specify only one of num_bins and bins
    
    Arguments:
        model {src.methods.base.mode.Model} -- A base model to compute the calibration for.
        dataset {torch.utils.data.Dataset} -- A dataset to compute the plot for.
        num_bins {int} -- The number of bins to split the data into. Will evenly split the data
        based on confidence.
        bin_edges {np.array} -- Specified edges of a series of bins. Will be used if not None
    """

    if (bin_edges is None and num_bins is None) or (bin_edges is not None and num_bins is not None):
        raise ValueError('Specify exactly one of bin_edges and num_bins')

    dataloader = DataLoader(
        dataset,
        batch_size=model.batch_size,
        num_workers=0,
        shuffle=False
    )

    pre_preds = None
    preds = None
    targets = dataset.targets

    for x, _ in dataloader:
        pre_pred, pred = model.predict(x)
        pre_pred, pred = pre_pred.cpu().numpy(), pred.cpu().numpy()

        if pre_preds is not None:
            pre_preds.cat((pre_preds, pre_pred), dim=0)
            preds = preds.cat((preds, pred), dim=0)

    # TODO: Check this is the correct thing to do
    # TODO: Think about epistemic vs aleotoric uncertainty?

    pre_preds.cpu()
    preds.cpu()

    mean_preds = preds.mean(dim=1)

    bin_edges = torch.tensor(bin_edges)

    num_bins = (bin_edges.shape) - 1

    mean_confidences = torch.zeros(num_bins)
    mean_accuracies = torch.zeros(num_bins)

    for i in range(num_bins):
        inds = (mean_preds > bin_edges[i]) and (mean_preds < bin_edges[i+1])
        preds = mean_preds[inds]
        targs = targets[inds]
        mean_acc = preds.eq(targs).mean()
        mean_conf = preds.mean()

        mean_confidences[i] = mean_conf
        mean_accuracies[i] = mean_acc

    return bin_edges.numpy(), mean_confidences.numpy(), mean_accuracies.numpy()

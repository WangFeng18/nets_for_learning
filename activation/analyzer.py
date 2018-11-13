import torch
import torchvision
import numpy

class Neuron:
    def __init__(self, act_topk_imgs, act_topk_labels, act_topk_values):
        self.act_topk_imgs = act_topk_imgs
        self.act_topk_labels = act_topk_labels
        self.act_topk_values = act_topk_values
    def get_grid(self):
        torchvision.utils.save_image(torch.from_numpy(self.act_topk_imgs), filename='test.png', nrow=10)

class Analyzer(object):
    def __init__(self, network, dataloader):
        self.network = network
        self.network.cuda()
        self.network.eval()
        self.dataloader = dataloader
    def forward(self):
        all_image = torch.zeros(0, dtype=torch.float)
        all_label = torch.zeros(0, dtype=torch.long)
        all_indicator = torch.zeros(0)
        for i_batch, data in enumerate(self.dataloader):
            label = data[1].cuda()
            input = data[0].cuda()
            all_label = torch.cat([all_label, data[1]], 0)
            all_image = torch.cat([all_image, data[0]], 0)
            output = self.network(input)
            act_outputs = self.network.act_outputs
            # convolution output should be b x oc x w x h
            act_indicator = act_outputs.max(3)[0].max(2)[0]
            all_indicator = torch.cat([all_indicator, act_indicator.cpu().detach()], 0)
            print i_batch
        all_indicator = all_indicator.numpy().transpose((1,0))
        all_image = all_image.numpy()
        all_label = all_label.numpy()
        indices = all_indicator.argsort(axis=1)
        print indices
        topk = 20
        all_neurons = []
        for neuron_id in range(indices.shape[0]):
            topk_indices = indices[neuron_id][::-1][:topk]
            print topk_indices
            act_topk_imgs = all_image[topk_indices]
            act_topk_labels = all_label[topk_indices]
            act_topk_values = all_indicator[neuron_id][topk_indices]
            current_neuron = Neuron(act_topk_imgs, act_topk_labels, act_topk_values)
            current_neuron.get_grid()
            all_neurons.append(current_neuron)
        return all_neurons

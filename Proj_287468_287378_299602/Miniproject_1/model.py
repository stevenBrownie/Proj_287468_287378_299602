import torch
from torch import nn
from torch import optim
from pathlib import Path


best_model_path=str(Path(__file__).parent)+'/bestmodel.pth'
class Model():
    def __init__(self) -> None:
        ## instantiate model + optimizer + loss function
        self.model = nn.Sequential(
            # encoder
            nn.Conv2d(3, 64, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 512, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),

            # decoder
            nn.ConvTranspose2d(512, 256, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(2, 2), stride=(1, 1), padding=1, dilation=1, bias=True)
        )
        # loss function used
        self.criterion = nn.MSELoss()

        # Optimizer
        self.learning_rate = 0.05
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.learning_rate, max_iter=20)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)#, betas=(0.9, 0.999), eps=1e-08)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)  # momAGD
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,momentum=0.9,nesterov=True) #nesterov
        # self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.learning_rate)

        # get the history of the loss during the training
        self.loss_history = None

        # use GPU instead of CPU for faster training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda = torch.cuda.is_available()

        # init weight
        self._init_weights()

    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for m in self.model:
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data = (nn.init.xavier_normal_(m.weight.data) > 0) * nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        path=best_model_path
        if self.use_cuda:
            self.model.load_state_dict(torch.load(path)) 
        else:
            self.model.load_state_dict(
                torch.load(path, map_location='cpu')) 

    def save_model(self) -> None:
        torch.save(self.model.state_dict(), best_model_path)

    def train(self, train_input, train_target, num_epochs=1, mb_size=4, print_evolution=False, hybrid=False) -> None:
        '''
        train_input:      tensor of size (N, C, H, W) containing a noisy version of the images.
        train_target:     tensor of size (N, C, H, W) containing another noisy version of the
                          same images, which only differs from the input by their noise.
        num_epochs:        number of epochs to train the model
        mb_size:          minibatch size
        print_evolution:  bool, if True, the loss and the number of epoch will be printed during
                          the training
        hybrid:           bool, if True, the hybrid strategie will be used for the optimizer
        '''
        # if the data in in range: 0-255, we normalize them
        if train_input.type()=='torch.ByteTensor':
            train_input=train_input.float()/255.
            print('in if')
        if train_target.type()=='torch.ByteTensor':
            train_target = train_target.float()/255.

        # move data to GPU if it exists
        self.model.to(self.device)
        self.criterion.to(self.device)
        train_input = train_input.to(self.device)
        train_target = train_target.to(self.device)

        self.loss_history = train_input.new_zeros(
            num_epochs) 

        i = 0
        if hybrid == True:
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.learning_rate, max_iter=20)

        for e in range(num_epochs):
            for b in range(0, train_input.size(0), mb_size):

                # if LBFGS, need to define a closure

                def closure():
                    output = self.model(train_input[b:b + mb_size])
                    loss = self.criterion(output, train_target[b:b + mb_size])
                    self.optimizer.zero_grad()
                    loss.backward()
                    return loss

                self.optimizer.step(closure)

                i += 1
                if i == 35 and hybrid == True:
                    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

            loss = closure()
            self.loss_history[e] = loss
            if print_evolution:
                print("======> epoch: {}/{}, Loss:{}".format(e + 1, num_epochs, loss))

    def predict(self, test_input) -> torch.Tensor:
        '''
        test_input: tensor of size (N1, C, H, W) that has to be denoised by the trained
                    or the loaded network
        return:     tensor of the size (N1, C, H, W)
        '''
        if self.use_cuda:
            test_input = test_input.cuda()
            self.model.cuda()
    
        if test_input.type()=='torch.ByteTensor':
            test_input=test_input.float()/255.
       
        predicted_tensor = self.model(test_input) * 255.
        return (predicted_tensor.detach().cpu() > 0) * predicted_tensor.detach().cpu() + 1e-13  # we want positive prediction
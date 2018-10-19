import track

from dataset import build_dataset, num_classes
from models import build_model
from optimizers import build_optimizer


def add_train_args(parser):
    # Main arguments go here
    parser.add_argument('--log_interval', default=10,
                        help='frequency (in iters) of logging')
    parser.add_argument('--arch', default='resnet50')
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--lr', default=.1, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=200, type=int)


def _train(epoch, loader, model, optimizer):
    raise NotImplementedError()


def _test(epoch, loader, model):
    raise NotImplementedError()


def do_training(args):
    trainloader, testloader = build_dataset(args.dataset,
                                            dataroot=args.dataroot,
                                            batch_size=args.batch_size,
                                            eval_batch_size=args.eval_batch_size,
                                            num_workers=2)
    model = build_model(args.arch, num_classes=num_classes(args.dataset))
    optimizer = build_optimizer('SGD', params=model.parameters(), lr=args.lr)
    # For example... 
    for epoch in range(args.epochs):
        track.debug("Starting epoch %d" % epoch)
        _train(epoch, trainloader, model, optimizer)
        _test(epoch, testloader, model)

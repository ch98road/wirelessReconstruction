# from torchvision.utils import save_image
# import os
# output = ()
# save_image(output.data,
#            os.path.join('./output/', '{}_fake.png'.format(iteration + 1)))


# from tensorboardX import SummaryWriter
# writer = SummaryWriter('./log/sngan/')
# max_score = 0

# for iteration in range(epoch):
#     g1_loss = criten(input, output)
#     g2_loss= criten(input, output)

#     info = {
#         'g1_loss': g1_loss.item(),
#         'loss': g2_loss.item(),
#     }
#     for tag, value in info.items():

#         writer.add_scalar(tag, value, iteration + 1)
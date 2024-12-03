if __name__ == "__main__":
    from src.ecolaf import *
    from torchvision import models
    import matplotlib.pyplot as plt
    import numpy as np

    m1 = models.segmentation.deeplabv3_resnet50(num_classes=21)
    m2 = models.segmentation.deeplabv3_resnet50(num_classes=21)
    m3 = models.segmentation.deeplabv3_resnet50(num_classes=21)
    m4 = models.segmentation.deeplabv3_resnet50(num_classes=21)

    img1 = torch.rand(4, 3, 500, 500)
    img2 = torch.rand(4, 3, 500, 500)
    img3 = torch.rand(4, 3, 500, 500)
    img4 = torch.rand(4, 3, 500, 500)

    m = ECOLAF([m1, m2, m3, m4], num_classes=20)

    out, out_u, conf_map, discounting_map = m([img1, img2, img3, img4], output_unimodal=True, output_conflict=True, output_discounting_coef=True, output_keyword='out')
    conf_map = conf_map.detach()
    discounting_map = discounting_map.detach()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(conf_map[0, 0, ...], cmap='viridis')
    axs[0, 0].set_title('conflict 1')

    axs[0, 1].imshow(conf_map[0, 1, ...], cmap='viridis')
    axs[0, 1].set_title('Conflict 2')

    axs[1, 0].imshow(conf_map[0, 2, ...], cmap='viridis')
    axs[1, 0].set_title('Conflict 3')

    axs[1, 1].imshow(conf_map[0, 3, ...], cmap='viridis')
    axs[1, 1].set_title('Conflict 4')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(discounting_map[0, 0, ...], cmap='viridis')
    axs[0, 0].set_title('Discounting 1')

    axs[0, 1].imshow(discounting_map[0, 1, ...], cmap='viridis')
    axs[0, 1].set_title('Discounting 2')

    axs[1, 0].imshow(discounting_map[0, 2, ...], cmap='viridis')
    axs[1, 0].set_title('Discounting 3')

    axs[1, 1].imshow(discounting_map[0, 3, ...], cmap='viridis')
    axs[1, 1].set_title('Discounting 4')

    plt.tight_layout()
    plt.show()

    pred = out[0].data.cpu().numpy()
    pred = np.argmax(pred, axis=0)

    pred1 = out_u[0][0].data.cpu().numpy()
    pred1 = np.argmax(pred1, axis=0)

    pred2 = out_u[1][0].data.cpu().numpy()
    pred2 = np.argmax(pred2, axis=0)

    pred3 = out_u[2][0].data.cpu().numpy()
    pred3 = np.argmax(pred3, axis=0)

    pred4 = out_u[3][0].data.cpu().numpy()
    pred4 = np.argmax(pred4, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(pred1, cmap='viridis')
    axs[0, 0].set_title('Prediction 1')

    axs[0, 1].imshow(pred2, cmap='viridis')
    axs[0, 1].set_title('Prediction 2')

    axs[1, 0].imshow(pred3, cmap='viridis')
    axs[1, 0].set_title('Prediction 3')

    axs[1, 1].imshow(pred4, cmap='viridis')
    axs[1, 1].set_title('Prediction 4')

    plt.tight_layout()
    plt.show()


    plt.imshow(pred)
    plt.title('fusion prediction')
    plt.show()

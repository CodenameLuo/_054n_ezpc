import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

# 改用带交叉熵损失的修改版（forward 多一个真标签入参 y）
from model_my_modified import EZPC
from utils import (
    DATASET_CHOICES,
    set_seed,
    load_backbone,
    backbone_to_name,
    load_dataset_config,
    load_train_embeddings,
    load_train_labels,
    load_class_split,
    get_text_embs,
    plot_loss
)

def optimize_param(ezpc_model, dataloader, seen_class_name_embs, num_epochs, lr, device):
    optimizer = Adam(ezpc_model.parameters(), lr=lr)

    total_loss_list = []
    matching_loss_list, ce_loss_list = [], []

    best_A = None
    best_loss = {'matching_loss': None, 'ce_loss': None, 'total_loss': None}

    for epoch in range(num_epochs):
        epoch_matching_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_total_loss = 0.0

        for emb_batch, label_batch in dataloader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            # ty 传 seen 类名嵌入、label_batch 是 seen 局部下标 → 交叉熵只在 seen 列上算
            matching_loss, ce_loss, total_loss = ezpc_model(emb_batch, seen_class_name_embs, label_batch)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Normalize A columns
            ezpc_model.normalize_weights()

            epoch_matching_loss += matching_loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_total_loss += total_loss.item()

        n_batches = len(dataloader)
        epoch_matching_loss /= n_batches
        epoch_ce_loss /= n_batches
        epoch_total_loss /= n_batches

        matching_loss_list.append(epoch_matching_loss)
        ce_loss_list.append(epoch_ce_loss)
        total_loss_list.append(epoch_total_loss)

        if best_loss['total_loss'] is None or epoch_total_loss < best_loss['total_loss']:
            best_A = ezpc_model.A.clone().detach()
            best_loss.update({
                'matching_loss': epoch_matching_loss,
                'ce_loss': epoch_ce_loss,
                'total_loss': epoch_total_loss
            })

        if epoch % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Matching: {epoch_matching_loss:.6f} | "
                f"CE: {epoch_ce_loss:.6f} | "
                f"Total: {epoch_total_loss:.6f}", flush=True)

    print(f"Final Matching Loss: {matching_loss_list[-1]}")
    print(f"Final CE Loss: {ce_loss_list[-1]}")
    print(f"Final Total Loss: {total_loss_list[-1]}")

    print(f"Best Matching Loss: {best_loss['matching_loss']}")
    print(f"Best CE Loss: {best_loss['ce_loss']}")
    print(f"Best Total Loss: {best_loss['total_loss']}")

    return best_A, matching_loss_list, ce_loss_list, total_loss_list

def main(args):
    # Create a dir to save the checkpoint
    backbone_name = backbone_to_name(args.backbone)
    output_folder = (
        f"{args.output_path}/{args.dataset}_backbone_{backbone_name}"
        f"_weight_{args.lambda_weight}_epoch_{args.num_epochs}"
        f"_lr_{args.lr}_bs_{args.batch_size}"
    )
    os.makedirs(output_folder, exist_ok=True)

    # Load the CLIP/SigLIP backbone
    model, _, tokenizer = load_backbone(args.backbone, args.device)

    # Get the classnames and concepts
    classnames, concept_names = load_dataset_config(args.dataset_root, args.dataset)

    # 加载训练图像嵌入 + 对应真实标签（标签是全类空间 id，与嵌入逐行对齐）
    train_img_tensor = load_train_embeddings(args.dataset_root, args.dataset, args.backbone)
    train_labels = load_train_labels(args.dataset_root, args.dataset)

    # 只在 seen 类列上做 CE：取 seen 类 id，把全局标签重映射成 seen 局部下标 0..K_seen-1
    # （与 test.py 受限口径一致：seen_map[global_id] = local_idx）
    class_split = load_class_split(args.dataset_root, args.dataset)
    seen_class_ids = class_split['seen_classes']
    seen_map = {global_id: local_idx for local_idx, global_id in enumerate(seen_class_ids)}
    train_labels = torch.tensor([seen_map[int(g)] for g in train_labels], dtype=torch.long)

    # dataloader 现在每批吐 (图像嵌入, seen 局部标签)
    dataloader = DataLoader(
        TensorDataset(train_img_tensor, train_labels),
        batch_size=args.batch_size, shuffle=True, pin_memory=False
    )

    # Generate classname and concept embeddings
    class_name_embs = get_text_embs(model, classnames, args.backbone, tokenizer, device=args.device)
    concept_matrix = get_text_embs(model, concept_names, args.backbone, tokenizer, device=args.device)

    # 只取 seen 类的类名嵌入当 CE 的目标列（K_seen 列，与 seen 局部标签一一对应）
    seen_class_name_embs = class_name_embs[seen_class_ids]

    # Init EZPC
    ezpc_model = EZPC(concept_matrix, args.lambda_weight).to(args.device)

    # Optimize params
    best_A, matching_loss_list, ce_loss_list, total_loss_list = \
        optimize_param(ezpc_model, dataloader, seen_class_name_embs, args.num_epochs, args.lr, args.device)

    # Save the weights
    torch.save(best_A, f"{output_folder}/best_A.pth")

    # Save the loss plots
    last_n_epochs = min(8000, args.num_epochs)
    start_epoch = args.num_epochs - last_n_epochs
    plot_loss(
        output_folder,
        matching_loss_list[start_epoch:],
        ce_loss_list[start_epoch:],
        total_loss_list[start_epoch:],
        start_epoch
    )

if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(1234)

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--num_epochs", type=int, default=10000, 
                        help="Number of Epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--lambda_weight", type=float, default=1.0,
                        help="Lambda weight (reconstruction loss coefficient)")
    parser.add_argument("--backbone", type=str, default="RN50",
                        help="CLIP/SigLIP backbone (e.g. RN50, ViT-B/32, ViT-L/14, siglip-so400m-patch14-384)")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASET_CHOICES,
                        help="Dataset name")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to the root dataset folder")
    parser.add_argument("--batch_size", type=int, default=1000000, 
                        help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    parser.add_argument("--output_path", type=str, default="./checkpoints", 
                        help="Directory for checkpoints")

    args = parser.parse_args()

    # Run the main function
    main(args)
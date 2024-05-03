from tqdm import tqdm
import torch

def train_fn(model, dataloader, optimizer, criterion, device):

    model.train() # ON Dropout
    total_loss = 0.0

    for A, P, N in tqdm(dataloader):
        A, P, N = A.to(device), P.to(device), N.to(device)

        A_embs = model(A)
        P_embs = model(P)
        N_embs = model(N)

        loss = criterion(A_embs, P_embs, N_embs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_fn(model, dataloader, criterion, device):

    model.eval() # OFF Dropout
    total_loss = 0.0

    with torch.no_grad():
        for A, P, N in tqdm(dataloader):
            A, P, N = A.to(device), P.to(device), N.to(device)

            A_embs = model(A)
            P_embs = model(P)
            N_embs = model(N)

            loss = criterion(A_embs, P_embs, N_embs)

            total_loss += loss.item()


    return total_loss / len(dataloader)


def train_fn_plus(model, dataloader, optimizer, criterion, device):

    model.train() # ON Dropout
    total_loss = 0.0

    for A_text, A_meta, P_text, P_meta, N_text, N_meta in tqdm(dataloader):
        
        A_text, A_meta, P_text, P_meta, N_text, N_meta = A_text.to(device), A_meta.to(device), P_text.to(device), P_meta.to(device), N_text.to(device), N_meta.to(device)

        A_embs = model(A_text, A_meta)
        P_embs = model(P_text, P_meta)
        N_embs = model(N_text, N_meta)

        loss = criterion(A_embs, P_embs, N_embs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def eval_fn_plus(model, dataloader, criterion, device):

    model.eval() # OFF Dropout
    total_loss = 0.0

    with torch.no_grad():
        for A_text, A_meta, P_text, P_meta, N_text, N_meta in tqdm(dataloader):
            
            A_text, A_meta, P_text, P_meta, N_text, N_meta = A_text.to(device), A_meta.to(device), P_text.to(device), P_meta.to(device), N_text.to(device), N_meta.to(device)

            A_embs = model(A_text, A_meta)
            P_embs = model(P_text, P_meta)
            N_embs = model(N_text, N_meta)

            loss = criterion(A_embs, P_embs, N_embs)

            total_loss += loss.item()


    return total_loss / len(dataloader)

# Training function for contrastive loss
def train_fn_contrastive(model, dataloader, optimizer, criterion, device):

    model.train() # ON Dropout
    total_loss = 0.0

    for A, P, N in tqdm(dataloader):
        A, P, N = A.to(device), P.to(device), N.to(device)

        A_embs = model(A)
        P_embs = model(P)
        N_embs = model(N)

        ones_like_A_embs = torch.ones_like(A_embs)

        loss_positive = criterion(A_embs, P_embs, target[0])
        loss_negative = criterion(A_embs, N_embs, target[1])

        # Compute total loss
        loss = loss_positive + loss_negative

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Training function for contrastive loss
def eval_fn_contrastive(model, dataloader, criterion, device):

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for A, P, N in tqdm(dataloader):
            
            A, P, N = A.to(device), P.to(device), N.to(device)

            A_embs = model(A)
            P_embs = model(P)
            N_embs = model(N)

          
            target = torch.tensor([1.0, -1.0]).to(device)

            loss_positive = criterion(A_embs, P_embs, target[0])
            loss_negative = criterion(A_embs, N_embs, target[1])

            # Compute total loss
            loss = loss_positive + loss_negative

            total_loss += loss.item()

    return total_loss / len(dataloader)
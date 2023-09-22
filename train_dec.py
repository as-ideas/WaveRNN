import torch
import torch.nn as nn
import torch.optim as optim

# Define the Transformer Decoder
class TransformerDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TransformerDecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, trg, memory):
        trg = self.embedding(trg)
        output = self.transformer_decoder(trg, memory)
        output = self.fc(output)
        return output

# Prepare toy data
# Replace this with your own data loading and preprocessing
vocab_size = 1000  # Replace with the actual size of your vocabulary
sequence_length = 20
batch_size = 32

src_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))
trg_sequence = torch.randint(0, vocab_size, (batch_size, sequence_length))

# Initialize the model
d_model = 256
nhead = 4
num_layers = 4

model = TransformerDecoderModel(vocab_size, d_model, nhead, num_layers)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(src_sequence, trg_sequence)  # Pass None for memory as it's a decoder-only model
    output = output.view(-1, vocab_size)
    loss = criterion(output, trg_sequence)
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from utilities import Utilities
from classifier import NN1DAN
import globals
import torch.optim as optim

from transformer import DecoderModel, EncoderModel
from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :globals.block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, globals.block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(globals.device), Y.to(globals.device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(globals.device), Y.to(globals.device)
        _, loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        # total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    
    # TODO: change back
    # Set up argument parser
    # part = int(getPartFromArgParser())
    # print(f'Part: {part}')
    part = 2
    
    # Load dataset
    print("Loading data and creating tokenizer ...")
    # texts = load_texts('speechesdataset')
    # load all sentences with labels
    texts = load_texts('./PA2_code/speechesdataset')
    
    # create a simple tokenizer - each unique word has an embedding in a dictionary
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the datasets
    print("Vocabulary size is", tokenizer.vocab_size)

    # Part 1: Classifier
    if part == 1:
        # training dataset
        # convert the whole dataset into indices (encoding)
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "./PA2_code/speechesdataset/train_CLS.tsv")
        # split the encoded dataset (indices) into batches
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=globals.batch_size, collate_fn=collate_batch,shuffle=True)
        
        # test dataset
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "./PA2_code/speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=globals.batch_size, collate_fn=collate_batch,shuffle=True)

        classifier = NN1DAN(input_size = globals.n_input, tokenizer = tokenizer)

        # for the classification  task, you will train for a fixed number of epochs like this:
        # loss function to be used in training
        criterion = torch.nn.NLLLoss()
        # used to update the weights
        optimizer = optim.Adam(list(classifier.parameters()), lr=globals.learning_rate)
        print(f'number of classifier parameters: {len(list(classifier.parameters()))}')

        for epoch in range(globals.epochs_CLS):
            # i = 0
            epoch_loss = 0.0
            for xb, yb in train_CLS_loader:
                # print(i + 1)
                # i += 1
                # print(f'xb_shape: {xb.shape}')
                # print(f'yb_shape: {yb.shape}')
                xb, yb = xb.to(globals.device), yb.to(globals.device)
                
                # CLS training code here
                # Zero the parameter gradients
                optimizer.zero_grad() # reset gradients to 0.0

                outputs, attn_maps = classifier(xb)
                
                # Both components are trained simultaneously, enabling the encoder to learn representations that are specifically useful for the speech segment classification task.
                # Compute loss and backpropagate
                loss = criterion(outputs, yb) # compare predictions to labels to compute the loss according to the selected loss fn = NLL
                loss.backward() # -> compute gradients
                optimizer.step() # -> update weights of both encoder and classifier at once - they are in the same object (inheriting from nn.Module)

                epoch_loss += loss.item()

            # Logging the loss and accuracy for each epoch
            # TODO: correct implementation?
            training_acc = compute_classifier_accuracy(classifier, train_CLS_loader)
            test_acc = compute_classifier_accuracy(classifier, test_CLS_loader)
            
            print(f'Epoch [{epoch+1}/{globals.epochs_CLS}], Loss: {epoch_loss / len(train_CLS_loader):.4f}, Training accuracy: {training_acc:.4f},  Test accuracy: {test_acc:.4f}')
            # break

        runSanityChecks(tokenizer, classifier, train_CLS_dataset)
        
    # Part 2: Decoder 
    elif part == 2:
        # training
        train_LM_dataset, train_LM_loader = createDatasetAndLoader(input_file = "./PA2_code/speechesdataset/train_LM.txt", tokenizer = tokenizer)
        
        # test
        test_LM_obama_dataset, test_LM_obama_loader = createDatasetAndLoader(input_file = "./PA2_code/speechesdataset/test_LM_obama.txt", tokenizer = tokenizer)
        test_LM_wbush_dataset, test_LM_wbush_loader = createDatasetAndLoader(input_file = "./PA2_code/speechesdataset/test_LM_wbush.txt", tokenizer = tokenizer)
        test_LM_hbush_dataset, test_LM_hbush_loader = createDatasetAndLoader(input_file = "./PA2_code/speechesdataset/test_LM_hbush.txt", tokenizer = tokenizer)
        
        print('Done creating all datasets and loaders')
        
        decoder = DecoderModel(vocab_size = tokenizer.vocab_size, n_embd = globals.n_embd, block_size = globals.block_size, num_heads = globals.n_head, num_layers = globals.n_layer)
        # should this be cross entropy?
        criterion = torch.nn.CrossEntropyLoss()
        # used to update the weights
        optimizer = optim.Adam(list(decoder.parameters()), lr=globals.learning_rate)
        
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            # print(f'xb shape: {xb.shape}')
            # print(f'yb shape: {yb.shape}')
            if i >= globals.max_iters:
                break
            xb, yb = xb.to(globals.device), yb.to(globals.device)
            # LM training code here
            optimizer.zero_grad() # reset gradients to 0.0
            outputs, loss, attn_maps = decoder(xb, yb)
            
            # print('DECODER OUTPUT')
            # print(outputs.shape)
            
            # print('ACTUAL OUTPUT')
            # print(yb.shape)
            
            # loss = criterion(outputs, yb) # compare predictions to labels to compute the loss according to the selected loss fn = NLL
            loss.backward() # -> compute gradients
            optimizer.step() # -> update weights of both encoder and classifier at once - they are in the same object (inheriting from nn.Module)
            # break
            
            if (i + 1)%100 == 0 or i == 499: 
                print(f'Training: Perpelxity after {i + 1} iterations: {compute_perplexity(decoder, train_LM_loader):.4f}')
                print('\n')
        
        print(f'Test (Obama): Perpelxity after {i} iterations: {compute_perplexity(decoder, test_LM_obama_loader):.4f}')
        print(f'Test (W Bush): Perpelxity after {i} iterations: {compute_perplexity(decoder, test_LM_wbush_loader):.4f}')
        print(f'Test (H Bush): Perpelxity after {i} iterations: {compute_perplexity(decoder, test_LM_hbush_loader):.4f}')
        # runSanityChecks(tokenizer, decoder, train_LM_dataset)

            
            

def getPartFromArgParser(): 
    parser = argparse.ArgumentParser(description='Run section based on specified assignment part')
    parser.add_argument('--part', type=str, required=True, help='PA2 Section to run')
    args = parser.parse_args()
    part = args.part
    return part

def createDatasetAndLoader(input_file, tokenizer):
    # Read file
    with open(input_file, 'r', encoding='utf-8') as f:
        sourceText = f.read()
        
    # convert the whole dataset into indices (encoding)
    dataset = LanguageModelingDataset(tokenizer, sourceText, globals.block_size)
    # split the encoded dataset (indices) into batches
    loader = DataLoader(dataset, batch_size = globals.batch_size, shuffle=True)
    
    return dataset, loader

def runSanityChecks(tokenizer, model, dataset):
    # sanity checks
    utils = Utilities(tokenizer, model)
    # sentence 1
    sen = dataset.samples[1][1]
    print(sen)
    utils.sanity_check(sen, globals.block_size)
    
    # sentence 2
    sen = dataset.samples[2][1]
    print(sen)
    utils.sanity_check(sen, globals.block_size)

if __name__ == "__main__":
    main()

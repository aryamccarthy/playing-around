"""
How to optionally load in a trained model in PyTorch.
"""
# in training loop:

state = {'iter_num': iter_num,
     'enc_state': encoder.state_dict(),
     'dec_state': decoder.state_dict(),
     'opt_state': optimizer.state_dict(),
     'src_vocab': src_vocab,
     'tgt_vocab': tgt_vocab,
     }
filename = 'state_%010d.pt' % iter_num
torch.save(state, filename)



# prior to training loop:
if args.load_checkpoint is not None:
    state = torch.load(args.load_checkpoint[0])
    iter_num = state['iter_num']
    src_vocab = state['src_vocab']
    tgt_vocab = state['tgt_vocab']
else:
    iter_num = 0
    src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                       args.tgt_lang,
                                       args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

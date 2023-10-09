import torch


def test_docvqa_collator_output(docvqa_collator, docvqa_train_batch):
    batch_output = docvqa_collator(docvqa_train_batch)
    assert all(k in batch_output.keys() for k in ('image', 'label', 'text_target'))
    assert batch_output['image'].shape == torch.Size([4, 1, 448, 576])
    assert batch_output['label'].shape == torch.Size([4, 511])
    assert batch_output['text_target'].shape == torch.Size([4, 511])
    assert torch.equal(batch_output['label'][0][1:52], batch_output['text_target'][0][0:51])


def test_cord_collator_output(cord_collator, cord_train_batch):
    batch_output = cord_collator(cord_train_batch)
    assert all(k in batch_output.keys() for k in ('image', 'label', 'text_target'))
    assert batch_output['image'].shape == torch.Size([4, 1, 1478, 1108])
    assert batch_output['label'].shape == torch.Size([4, 511])
    assert batch_output['text_target'].shape == torch.Size([4, 511])
    assert torch.equal(batch_output['label'][0][1:57], batch_output['text_target'][0][0:56])


def test_rvlcdip_collator_output(rvlcdip_collator, rvlcdip_train_batch):
    batch_output = rvlcdip_collator(rvlcdip_train_batch)
    assert all(k in batch_output.keys() for k in ('image', 'label', 'text_target'))
    assert batch_output['image'].shape == torch.Size([4, 1, 448, 576])
    assert batch_output['label'].shape == torch.Size([4, 5])
    assert batch_output['text_target'].shape == torch.Size([4, 5])
    assert torch.equal(batch_output['label'][0][1:3], batch_output['text_target'][0][0:2])

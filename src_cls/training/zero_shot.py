import logging

import torch
from tqdm import tqdm
import ipdb
from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, IMAGENET_R_CLASSNAMES, IMAGENET_A_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from .precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=args.device, dtype=input_dtype)
            target = target.to(args.device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]  #(bs*cls_num,512)
                if args.cls_num == 1:
                    logits = 100. * image_features @ classifier  # classifier: (512,1000)
                else:
                    image_features = image_features.reshape(-1, args.cls_num, image_features.shape[-1]).transpose(0, 1)  # (cls_num,bs,512)
                    logits = 100. * image_features @ classifier
                    logits = torch.mean(logits, dim=0)

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision)
    with autocast():
        if 'imagenet-val' in data or 'imagenet-v2' in data:
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )
        if 'imagenet-r' in data:
            classifier_r = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_R_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )            
        if 'imagenet-a' in data:
            classifier_a = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_A_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=args.device,
                use_tqdm=True,
            )
    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5
    if 'imagenet-r' in data:
        top1, top5 = run(model, classifier_r, data['imagenet-r'].dataloader, args)
        results['imagenet-r-zeroshot-val-top1'] = top1
        results['imagenet-r-zeroshot-val-top5'] = top5
    if 'imagenet-a' in data:
        top1, top5 = run(model, classifier_a, data['imagenet-a'].dataloader, args)
        results['imagenet-a-zeroshot-val-top1'] = top1
        results['imagenet-a-zeroshot-val-top5'] = top5 
    logging.info('Finished zero-shot imagenet.')

    return results

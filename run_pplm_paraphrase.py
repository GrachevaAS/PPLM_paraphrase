#
#
#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from operator import add
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

SMALL_CONST = 1e-15
BIG_CONST = 1e10

REGULAR = 1
VERBOSE = 2
VERBOSITY_LEVELS = {
    'regular': REGULAR,
    'verbose': VERBOSE,
}


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        target_output,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        num_iterations=3,
        horizon_length=1,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR
):
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []

        ce_loss = torch.nn.CrossEntropyLoss()
        curr_unpert_past = unpert_past
        curr_probs = torch.unsqueeze(probs, dim=1)
        wte = model.resize_token_embeddings()
        for _ in range(horizon_length):
            inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
            _, curr_unpert_past, curr_all_hidden = model(
                past=curr_unpert_past,
                inputs_embeds=inputs_embeds
            )
            curr_hidden = curr_all_hidden[-1]
            new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                curr_hidden, dim=1)

        label = torch.tensor(curr_probs.shape[0] * [target_output],
                             device=device,
                             dtype=torch.long)
        discrim_loss = ce_loss(probs, label)
        if verbosity_level >= VERBOSE:
            print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
        loss += discrim_loss
        loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        loss.backward()

        grad_norms = [(torch.norm(p_.grad) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)]

        # normalize gradients
        grad = [
            -stepsize * (p_.grad / grad_norms[index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=False,
        num_iterations=3,
        horizon_length=1,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    grad_norms = None
    last = None
    loss_in_time = []

    target_output = output_so_far[:, 2:]
    output_so_far = output_so_far[:, :2]
    length = target_output.shape[-1]

    print(tokenizer.decode(output_so_far[:, 1:].tolist()[0]), end='')
    for i in range(length):
        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token
        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        if num_iterations == 0:
            pert_past = past
        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    target_output=target_output[:, i],
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=stepsize,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  
        pert_probs = F.softmax(pert_logits, dim=-1)

        # Fuse the modified model and original model
        unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
        pert_probs = ((pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale)))
        pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)
        if torch.sum(pert_probs) <= 1:
            pert_probs = pert_probs / torch.sum(pert_probs)
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(last), end='')

    return output_so_far, loss_in_time


def run_pplm_example(
        model,
        tokenizer,
        device,
        cond_text="",
        num_samples=1,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        horizon_length=1,
        gamma=1.0,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        verbosity='regular'
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)
    tokenized_cond_text = tokenizer.encode(
        tokenizer.bos_token + cond_text,
        add_special_tokens=False
    )

    if verbosity_level > REGULAR:
        print("= Phrase =")
        print(tokenizer.decode(tokenized_cond_text))
        print()

    if device == 'cuda':
        torch.cuda.empty_cache()
    pert_gen_tok_texts = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=tokenized_cond_text,
            device=device,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            horizon_length=horizon_length,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    generated_texts = []
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])
            print()
            if verbosity_level > REGULAR:
                print("= Perturbed generated text {} =".format(i + 1))
            print(pert_gen_text)
            print()
        except:
            pass
        generated_texts.append(pert_gen_tok_text)

    return generated_texts, losses_in_time

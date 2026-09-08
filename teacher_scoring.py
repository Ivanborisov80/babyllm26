import argparse
import random
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DebertaV2Tokenizer
from datasets import load_dataset


FEWSHOT = (
    'Rate how hard the missing piece is to predict from the visible text.\n'
    '0.0 = fully predictable, 1.0 = impossible to guess.\n\n'
    'Text: "the cat sat on the ___"\n'
    'Missing piece: "mat"\n'
    'Score: 0.15\n\n'
    'Text: "she was runn___"\n'
    'Missing piece: "ing"\n'
    'Score: 0.05\n\n'
    'Text: "she opened the door and saw a ___"\n'
    'Missing piece: "kangaroo"\n'
    'Score: 0.92\n\n'
    'Text: "it was an extra___ary"\n'
    'Missing piece: "ordin"\n'
    'Score: 0.55\n\n'
)


def build_prompt(visible_text, missing_piece):
    return (
        FEWSHOT
        + f'Text: "{visible_text}"\n'
        + f'Missing piece: "{missing_piece}"\n'
        + 'Score:'
    )


def sample_dataset(dataset, fraction, seed):
    random.seed(seed)
    n = len(dataset['train'])
    indices = random.sample(range(n), int(n * fraction))
    return dataset['train'].select(indices)


def build_token_tables(tokenizer, vocab_size):
    has_alpha = [False] * vocab_size
    starts_word = [False] * vocab_size
    piece_text = [''] * vocab_size
    for tid in range(vocab_size):
        piece = tokenizer.convert_ids_to_tokens(tid)
        if not isinstance(piece, str) or not piece:
            continue
        if piece.startswith('\u2581'):
            starts_word[tid] = True
            body = piece[1:]
        else:
            body = piece
            if not any(c.isalnum() for c in piece):
                if not all(c in "'\u2019-" for c in piece):
                    starts_word[tid] = True
        piece_text[tid] = body
        if body and any(c.isalpha() for c in body):
            has_alpha[tid] = True
    return has_alpha, starts_word, piece_text


def build_fragment_view(student_ids, p, starts_word, piece_text,
                        student_tokenizer, max_context_chars):
    n = len(student_ids)
    ws = p
    while ws > 0 and not starts_word[student_ids[ws]]:
        ws -= 1
    we = p + 1
    while we < n and not starts_word[student_ids[we]]:
        we += 1

    left_context = student_tokenizer.decode(student_ids[:ws]).strip()
    left_context = left_context[-max_context_chars:]

    prefix = ''.join(piece_text[student_ids[i]] for i in range(ws, p))
    suffix = ''.join(piece_text[student_ids[i]] for i in range(p + 1, we))
    missing = piece_text[student_ids[p]]

    word_with_gap = f"{prefix}___{suffix}"
    visible = (left_context + ' ' + word_with_gap).strip()
    return visible, missing


def pick_target_positions(student_ids, k, rng, counts_list=None, target_obs=20,
                          has_alpha=None):
    n = len(student_ids)
    if has_alpha is None:
        candidates = list(range(1, n))
    else:
        candidates = [p for p in range(1, n) if has_alpha[student_ids[p]]]
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates
    if counts_list is None:
        return rng.sample(candidates, k)
    scored = []
    for p in candidates:
        boost = 1.0 if counts_list[student_ids[p]] < target_obs else 0.0
        scored.append((boost + rng.random(), p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def parse_score(generated):
    tok = generated.strip().split()
    if not tok:
        return 0.5, False
    try:
        val = float(tok[0].rstrip('.').rstrip(','))
    except ValueError:
        return 0.5, False
    if not (0.0 <= val <= 1.0):
        return min(max(val, 0.0), 1.0), False
    return val, True


def compute_teacher_score(
    dataset_path,
    output_path,
    student_tokenizer_path,
    teacher_model_name,
    fraction,
    tokens_per_example,
    batch_size,
    max_context_chars,
    mlm_prob,
    target_obs,
    seed,
):
    rng = random.Random(seed)

    print(f"Loading dataset from {dataset_path} ...", flush=True)
    dataset = load_dataset('text', data_files={'train': dataset_path})
    subset = sample_dataset(dataset, fraction, seed)
    print(f"Scoring on {len(subset)} examples ({fraction*100:.2f}% of dataset)", flush=True)

    student_tokenizer = DebertaV2Tokenizer.from_pretrained(student_tokenizer_path)
    vocab_size = student_tokenizer.vocab_size
    print(f"Student vocab size: {vocab_size}", flush=True)

    has_alpha, starts_word, piece_text = build_token_tables(student_tokenizer, vocab_size)
    n_eligible = sum(has_alpha)
    print(f"Eligible target tokens (contain letters): {n_eligible}/{vocab_size} "
          f"({n_eligible/vocab_size:.1%}). Fragments are shown inside their word "
          f"with a gap; punctuation and digits are never scored.", flush=True)

    print(f"Loading teacher tokenizer from {teacher_model_name} ...", flush=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "left"

    print("Loading teacher model ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    teacher.eval()
    print("Teacher model loaded.", flush=True)

    tokenized = []
    for example in subset:
        text = example['text']
        if not text.strip():
            continue
        student_ids = student_tokenizer(text, add_special_tokens=False)['input_ids']
        if len(student_ids) < 2:
            continue
        tokenized.append(student_ids)

    print(f"Tokenized {len(tokenized)} usable examples", flush=True)

    difficulty_sum = torch.zeros(vocab_size)
    difficulty_count = torch.zeros(vocab_size)
    counts_list = [0] * vocab_size
    n_parsed = 0
    n_failed = 0
    n_prompts = 0
    n_skipped = 0
    t0 = time.time()

    sentence_batch = max(1, batch_size // tokens_per_example)
    n_chunks = (len(tokenized) + sentence_batch - 1) // sentence_batch

    for chunk_idx, start in enumerate(range(0, len(tokenized), sentence_batch)):
        chunk = tokenized[start:start + sentence_batch]

        batch_jobs = []
        for student_ids in chunk:
            positions = pick_target_positions(
                student_ids, tokens_per_example, rng,
                counts_list=counts_list, target_obs=target_obs,
                has_alpha=has_alpha,
            )
            if not positions:
                n_skipped += 1
                continue
            for pos in positions:
                tid = student_ids[pos]
                if not (0 <= tid < vocab_size):
                    continue
                visible, missing = build_fragment_view(
                    student_ids, pos, starts_word, piece_text,
                    student_tokenizer, max_context_chars,
                )
                if not missing.strip():
                    continue
                batch_jobs.append((tid, build_prompt(visible, missing)))

        if not batch_jobs:
            continue

        prompts = [p for _, p in batch_jobs]
        inputs = teacher_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda:0")

        with torch.no_grad():
            out = teacher.generate(
                **inputs,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=teacher_tokenizer.pad_token_id,
            )

        prompt_len = inputs['input_ids'].shape[1]
        decoded = teacher_tokenizer.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )

        for (tid, _), gen in zip(batch_jobs, decoded):
            score, ok = parse_score(gen)
            if ok:
                n_parsed += 1
            else:
                n_failed += 1
            difficulty_sum[tid] += score
            difficulty_count[tid] += 1
            counts_list[tid] += 1

        n_prompts += len(batch_jobs)
        if chunk_idx % 20 == 0:
            elapsed = time.time() - t0
            chunk_rate = (chunk_idx + 1) / max(elapsed, 1e-6)
            eta = (n_chunks - chunk_idx - 1) / max(chunk_rate, 1e-6)
            covered = (difficulty_count > 0).sum().item()
            print(f"  [{start+len(chunk)}/{len(tokenized)} sentences, {n_prompts} prompts] "
                  f"{n_prompts/max(elapsed,1e-6):.1f} prompts/s  ETA {eta/60:.1f} min  "
                  f"parse_fail={n_failed/(max(n_parsed+n_failed,1)):.1%}  "
                  f"coverage={covered}/{vocab_size} ({covered/vocab_size:.1%})", flush=True)

    print(f"\nTotal prompts scored: {n_prompts}", flush=True)

    total = n_parsed + n_failed
    print(f"\nParse success: {n_parsed}/{total} ({n_parsed/max(total,1):.1%})", flush=True)
    print(f"Parse failure: {n_failed}/{total} ({n_failed/max(total,1):.1%}) "
          f"-- these fell back to 0.5", flush=True)

    covered = int((difficulty_count > 0).sum().item())
    print(f"Sentences skipped (no eligible target): {n_skipped}", flush=True)
    print(f"Vocab coverage: {covered}/{vocab_size} tokens got >=1 observation "
          f"({covered/vocab_size:.1%})", flush=True)
    print(f"  of eligible tokens: {covered}/{n_eligible} "
          f"({covered/max(n_eligible,1):.1%})", flush=True)
    print(f"Tokens with >=5 observations: "
          f"{(difficulty_count >= 5).sum().item()}", flush=True)
    print(f"Tokens with >=20 observations: "
          f"{(difficulty_count >= 20).sum().item()}", flush=True)

    observed = difficulty_count > 0
    n_observed = int(observed.sum().item())

    teacher_score = torch.full((vocab_size,), float(mlm_prob))
    if n_observed > 0:
        raw = difficulty_sum[observed] / difficulty_count[observed]
        prior = raw.mean()
        k_shrink = 5.0
        cnt = difficulty_count[observed]
        smoothed = (difficulty_sum[observed] + k_shrink * prior) / (cnt + k_shrink)
        scaled = mlm_prob * smoothed / smoothed.mean()
        teacher_score[observed] = scaled
    teacher_score = teacher_score.clamp(min=0.005, max=min(1.0, 4.0 * mlm_prob))

    obs_vals = teacher_score[observed] if n_observed > 0 else teacher_score
    print(f"\n--- teacher_score diagnostics ---", flush=True)
    print(f"observed tokens: {n_observed}/{vocab_size}", flush=True)
    print(f"unique values (observed): {obs_vals.unique().numel()}", flush=True)
    print(f"std  (observed): {obs_vals.std().item():.6f}", flush=True)
    print(f"mean (observed): {obs_vals.mean().item():.6f}", flush=True)
    print(f"min  (observed): {obs_vals.min().item():.6f}", flush=True)
    print(f"max  (observed): {obs_vals.max().item():.6f}", flush=True)
    print(f"unobserved tokens pinned at mlm_prob={mlm_prob}", flush=True)
    print(f"at clamp floor: "
          f"{(teacher_score <= 0.005001).float().mean().item():.1%}", flush=True)

    torch.save(teacher_score, output_path)
    print(f"\nSaved teacher_score ({vocab_size} values) -> {output_path}", flush=True)
    return teacher_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="teacher_score.pt")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--tokens_per_example", type=int, default=3,
                        help="how many token positions to score per sentence")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_context_chars", type=int, default=300)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--target_obs", type=int, default=20,
                        help="observations per token before it stops being "
                             "prioritised for scoring")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    compute_teacher_score(
        dataset_path=args.train_data,
        output_path=args.output_path,
        student_tokenizer_path=args.tokenizer,
        teacher_model_name=args.teacher_model,
        fraction=args.fraction,
        tokens_per_example=args.tokens_per_example,
        batch_size=args.batch_size,
        max_context_chars=args.max_context_chars,
        mlm_prob=args.mlm_prob,
        target_obs=args.target_obs,
        seed=args.seed,
    )

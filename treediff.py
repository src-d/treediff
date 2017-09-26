import argparse
from collections import defaultdict
import ctypes
from glob import glob
import importlib
import json
import logging
from pprint import pformat
import sys

import farmhash
Node = importlib.import_module(
    "bblfsh.gopkg.in.bblfsh.sdk.v1.uast.generated_pb2").Node
from modelforge.logs import setup_logging
import numpy
import lapjv


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--hash-rounds", type=int, default=10)
    args = parser.parse_args()
    setup_logging("INFO")
    return args

HASH_SIZE = 16


def hash_self(node):
    roles = list(node.roles)
    seed1 = 0
    for i in range(min(4, len(roles))):
        seed1 |= roles[i] << (i << 3)
    seed2 = 0
    if len(roles) > 4:
        for i in range(min(4, len(roles) - 4)):
            seed2 |= roles[i + 4] << (i << 3)
    h1, h2 = farmhash.hash128withseed(node.token, seed1, seed2)
    return 1, h1.to_bytes(8, "little") + h2.to_bytes(8, "little")


def hash_node(node, seed, mapping, debug=False):
    children = list(node.children)
    if not children:
        h = hash_self(node)
        mapping[id(node)] = h[1]
        return h
    inner_hashes = [hash_node(c, seed, mapping) for c in children] + [hash_self(node)]
    weights = [h[0] for h in inner_hashes]
    total = sum(weights)
    sample_sizes = []
    for w in weights:
        s = max(1, (w * HASH_SIZE) // total)
        sample_sizes.append(s)
    delta = HASH_SIZE - sum(sample_sizes)
    if delta != 0:
        ss2 = [(s, i) for i, s in enumerate(sample_sizes)]
        ss2.sort(reverse=True)
        fix = 1 if delta > 0 else -1
        possible = True
        while delta != 0 and possible:
            possible = False
            for s, i in ss2:
                if sample_sizes[i] < 2 and fix < 0:
                    continue
                possible = True
                sample_sizes[i] += fix
                delta -= fix
                if delta == 0:
                    break
        if possible:
            for s in sample_sizes:
                assert s > 0
        else:
            # random sample 15 nodes
            indices = list(range(len(sample_sizes) - 1))
            numpy.random.seed(seed)
            choice = numpy.random.choice(indices, 15, replace=False)
            for i in range(len(sample_sizes)):
                sample_sizes[i] = 0
            for i in choice:
                sample_sizes[i] = 1
            sample_sizes[-1] = 1  # self
    choices = []
    if debug:
        print(weights, sample_sizes)
        print([h[1].hex() for h in inner_hashes])
    for h, s in zip(inner_hashes, sample_sizes):
        numpy.random.seed(seed)
        choice = numpy.random.choice(
            numpy.array(sorted(h[1]), dtype=numpy.uint8), s, replace=False)
        choices.append(choice)
    choices = numpy.hstack(choices)
    choices = choices.tobytes()
    mapping[id(node)] = choices
    return total, choices


def fingerprint_node(path, node, mapping):
    h = hash_self(node)[1]
    if path:
        mapping[id(node)] = path[-16:]
    else:
        mapping[id(node)] = None
    for child in node.children:
        fingerprint_node(path + h, child, mapping)


def dereference_idptr(value):
    return ctypes.cast(value, ctypes.py_object).value


def treediff(uast1, uast2, nseeds=10):
    log = logging.getLogger("treediff")
    dists = None
    supermap1 = defaultdict(bytes)
    supermap2 = defaultdict(bytes)
    for seed in range(nseeds):
        log.info("hash round %d/%d", seed + 1, nseeds)
        map1 = {}
        map2 = {}
        hash_node(uast1, seed, map1)
        hash_node(uast2, seed, map2)
        if dists is None:
            log.info("nodes before: %d", len(map1))
            log.info("nodes after: %d", len(map2))
            dists = numpy.ones((len(map1) + len(map2),) * 2, dtype=numpy.float32)
            dists *= 1024
            dists[:len(map1), len(map1):] = HASH_SIZE * nseeds + 1
            dists[len(map1):, :len(map1)] = HASH_SIZE * nseeds + 1
        byte_matches = [[] for _ in range(256)]
        for i, (k, h) in enumerate(map2.items()):
            supermap2[k] += h
            for b in set(h):
                byte_matches[b].append(i)
        for i, (k, h) in enumerate(map1.items()):
            supermap1[k] += h
            candidates = [0] * len(map2)
            for b in h:  # no set here!
                for j in byte_matches[b]:
                    candidates[j] -= 1
            dists[i, len(map1):] += candidates
            dists[len(map1):, i] += candidates

    """
    fingerprint_node(b"", uast1, map1)
    fingerprint_node(b"", uast2, map2)
    fingerprints = defaultdict(list)
    for i, h in enumerate(map2.values()):
        if h is not None:
            fingerprints[h].append(i)
    for i, h in enumerate(map1.values()):
        for j in fingerprints[h]:
            dists[i, j + len(map1)] -= 1
            dists[j + len(map1), i] -= 1
    """

    seq1 = list(map1)
    seq2 = list(map2)

    log.info("applying the offset hint")
    HIGHER_PRECISION_MAX_DIST = 2
    max_offset = 0
    for i in range(len(map1)):
        node = dereference_idptr(seq1[i])
        if node.start_position.line > 0:
            max_offset = max(node.end_position.offset, max_offset)
    for j in range(len(map2)):
        node = dereference_idptr(seq2[j])
        if node.start_position.line > 0:
            max_offset = max(node.end_position.offset, max_offset)
    for i, j in zip(*numpy.where(dists <= HIGHER_PRECISION_MAX_DIST)):
        if i >= len(map1):
            continue
        assert j >= len(map1)
        j -= len(map1)
        node_before = dereference_idptr(seq1[i])
        node_after = dereference_idptr(seq2[j])
        if node_before.start_position.line == 0 or node_after.start_position.line == 0:
            continue
        delta = abs(node_after.start_position.offset -
                    node_before.start_position.offset) / max_offset
        assert 0 <= delta < 1
        dists[i, len(map1) + j] += delta
        dists[len(map1) + j, i] += delta

    log.info("lapjv")
    row_ind, _, _ = lapjv.lapjv(dists)
    log.info("compiling edit script")
    threshold = HASH_SIZE / 2
    deleted = {(i, j) for (i, j) in enumerate(row_ind[:len(map1)])
               if j < len(map1) or dists[i, j] > threshold * nseeds}
    exact = {(i, j) for (i, j) in enumerate(row_ind[:len(map1)])
             if j >= len(map1) and supermap1[seq1[i]] == supermap2[seq2[j - len(map1)]]}
    mapped2 = {p[1] for p in (set(enumerate(row_ind[:len(map1)])) - deleted)}
    added = set(range(len(map1), len(row_ind))) - mapped2
    log.info("deleted: %d", len(deleted))
    log.info("added: %d", len(added))
    log.info("exact match: %d", len(exact))
    log.info("fuzzy match: %d", len(map1) - len(deleted) - len(exact))
    diff = []
    for i, _ in deleted:
        node = dereference_idptr(seq1[i])
        diff.append(("delete", node))
    for j in added:
        node = dereference_idptr(seq2[j - len(map1)])
        diff.append(("add", node))
    for i, j in set(enumerate(row_ind[:len(map1)])) - deleted - exact:
        diff.append(("modify", dereference_idptr(seq1[i]), dereference_idptr(seq2[j - len(map1)])))
    return diff


def write_diff(src_before, src_after, diff, output):
    log = logging.getLogger("treediff")
    DESCRIPTOR = importlib.import_module(
        "bblfsh.gopkg.in.bblfsh.sdk.v1.uast.generated_pb2").DESCRIPTOR
    Role = DESCRIPTOR.enum_types_by_name["Role"]
    file_role = Role.values_by_name["FILE"].number
    script = []
    for change in diff:
        action, node = change[:2]
        if file_role in node.roles:
            continue
        if not node.start_position.line:
            log.warning("node \"%s\" [%s] with %d children has no position, skipped",
                        node.token, ", ".join(Role.values_by_number[n].name for n in node.roles),
                        len(node.children))
            continue

        def format_position(pos):
            return {"line": pos.line, "col": pos.col, "offset": pos.offset}

        if action in ("add", "delete"):
            script.append((action, format_position(node.start_position),
                           format_position(node.end_position)))
        else:
            script.append((action, {
                "before": [format_position(change[1].start_position),
                           format_position(change[1].end_position)],
                "after": [format_position(change[2].start_position),
                          format_position(change[2].end_position)]}))
    log.info("SCRIPT:\n%s", pformat(script))
    with open(output, "w") as fout:
        json.dump({"src_before": src_before,
                   "src_after": src_after,
                   "script": script},
                  fout, sort_keys=True, indent=2)


def main():
    args = setup()
    base = args.sequence
    uast_before = glob("%s_before_*.pb" % base)[0]
    uast_after = glob("%s_after_*.pb" % base)[0]
    src_before = glob("%s_before_*.src" % base)[0]
    src_after = glob("%s_after_*.src" % base)[0]
    with open(uast_before, "rb") as fin:
        uast1 = Node.FromString(fin.read())
    with open(uast_after, "rb") as fin:
        uast2 = Node.FromString(fin.read())
    diff = treediff(uast1, uast2, nseeds=args.hash_rounds)
    with open(src_before) as fin:
        src_before = fin.read()
    with open(src_after) as fin:
        src_after = fin.read()
    write_diff(src_before, src_after, diff, args.output)


if __name__ == "__main__":
    sys.exit(main())

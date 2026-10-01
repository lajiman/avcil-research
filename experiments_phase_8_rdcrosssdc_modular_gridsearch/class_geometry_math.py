"""NumPy geometry on disjoint reference/query samples; no model or trainer."""
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prototypes(ref, labels, nclasses):
    ref = normalize(ref)
    counts = np.bincount(labels, minlength=nclasses)
    if len(counts) != nclasses or (counts == 0).any():
        raise ValueError('Reference must cover exactly all candidate classes')
    sums = np.zeros((nclasses, ref.shape[1]))
    np.add.at(sums, labels, ref)
    return normalize(sums), counts


def margin_scores(query, proto, labels, temperature=.1):
    if temperature <= 0 or len(proto) < 2:
        raise ValueError('Positive temperature and at least two classes required')
    scores = normalize(query) @ proto.T
    target = scores[np.arange(len(labels)), labels].copy()
    predicted = scores.argmax(1)
    scores[np.arange(len(labels)), labels] = -np.inf
    hard_negative = scores.argmax(1)
    hard_margin = target-scores.max(1)
    scaled = scores/temperature
    maximum = scaled.max(1)
    logsum = maximum+np.log(np.exp(scaled-maximum[:,None]).sum(1))
    return dict(hard_margin=hard_margin, logodds_margin=target/temperature-logsum,
                predicted=predicted, hard_negative=hard_negative, target_similarity=target)


def geometry(ref, ref_labels, query, query_labels, nclasses, k=5):
    ref, query = normalize(ref), normalize(query)
    proto, counts = prototypes(ref, ref_labels, nclasses)
    scores = margin_scores(query, proto, query_labels)
    k = min(k, len(ref))
    if k < 1:
        raise ValueError('k must be positive')
    purity = []
    for start in range(0, len(query), 128):
        similarities = query[start:start+128] @ ref.T
        # Stable tie handling follows the deterministic reference-ID order.
        neighbors = np.argsort(-similarities, axis=1, kind='stable')[:,:k]
        purity.extend((ref_labels[neighbors] == query_labels[start:start+128,None]).mean(1))
    purity = np.asarray(purity)
    proto_similarity = proto @ proto.T
    np.fill_diagonal(proto_similarity, -np.inf)
    rows = []
    for c in range(nclasses):
        mask = query_labels == c
        if not mask.any():
            raise ValueError('Query must cover all candidate classes')
        rows.append(dict(class_id=c, n_reference=int(counts[c]), n_query=int(mask.sum()),
            reference_dispersion=float((1-ref[ref_labels==c]@proto[c]).mean()),
            query_dispersion=float((1-scores['target_similarity'][mask]).mean()),
            mean_margin=float(scores['hard_margin'][mask].mean()),
            p10_margin=float(np.quantile(scores['hard_margin'][mask], .1)),
            centroid_accuracy=float((scores['predicted'][mask]==c).mean()),
            knn_purity=float(purity[mask].mean()),
            closest_class=int(proto_similarity[c].argmax()),
            closest_class_similarity=float(proto_similarity[c].max())))
    scores['knn_purity'] = purity
    return rows, scores, proto


def confusion_rows(labels, predicted, nclasses):
    matrix = np.zeros((nclasses,nclasses), dtype=np.int64)
    np.add.at(matrix, (labels, predicted), 1)
    rows = []
    for c in range(nclasses):
        tp = int(matrix[c,c]); support = int(matrix[c].sum()); fp = int(matrix[:,c].sum()-tp)
        rows.append(dict(class_id=c, support=support, tp=tp, fp=fp, fn=support-tp,
            precision=tp/(tp+fp) if tp+fp else 0., recall=tp/support if support else 0.,
            f1=2*tp/(support+tp+fp) if support+tp+fp else 0.))
    return rows, matrix


def retention(teacher_query, student_query, teacher_reference, ref_labels, labels,
              nclasses, temperature=.1, tolerance=.01):
    """Fixed TRAIN references / TEST queries: no LOO, no original-memory claim."""
    proto, _ = prototypes(teacher_reference, ref_labels, nclasses)
    before = margin_scores(teacher_query, proto, labels, temperature)['logodds_margin']
    after = margin_scores(student_query, proto, labels, temperature)['logodds_margin']
    drop = before-after
    violation = np.maximum(drop-tolerance, 0)
    return dict(reference_margin=before, current_margin=after, margin_drop=drop,
                violation=violation, active=violation>0)


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='stable')
    result = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start+1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        result[order[start:end]] = (start+end-1)/2
        start = end
    return result


def profile_correlation(a, b):
    a, b = rankdata(a), rankdata(b)
    if np.std(a)==0 or np.std(b)==0:
        return None
    return float(np.corrcoef(a,b)[0,1])

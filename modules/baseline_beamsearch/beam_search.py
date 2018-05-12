import numpy as np
import math
import collections

NEG_INF = -float("inf")


class beam_search():
    
  def __init__(self):
     pass

  def make_new_beam(self):
       fn = lambda : (NEG_INF, NEG_INF)
       return collections.defaultdict(fn)

  def logsumexp(self, *args):
      if all(a == NEG_INF for a in args):
         return NEG_INF
      a_max = max(args)
      lsp = math.log(sum(math.exp(a - a_max)
                      for a in args))
      return a_max + lsp

  def decode(self, probs, beam_size=100, blank=0):
     T, S = probs.shape
     probs = np.log(probs)
     beam = [(tuple(), (0.0, NEG_INF))]

     for t in range(T): # Loop over time

      # A default dictionary to store the next step candidates.
      next_beam = self.make_new_beam()

      for s in range(S): # Loop over vocab
        p = probs[t, s]
        for prefix, (p_b, p_nb) in beam: # Loop over beam

          if s == blank:
            n_p_b, n_p_nb = next_beam[prefix]
            n_p_b = self.logsumexp(n_p_b, p_b + p, p_nb + p)
            next_beam[prefix] = (n_p_b, n_p_nb)
            continue

          end_t = prefix[-1] if prefix else None
          n_prefix = prefix + (s,)
          n_p_b, n_p_nb = next_beam[n_prefix]
          if s != end_t:
            n_p_nb = self.logsumexp(n_p_nb, p_b + p, p_nb + p)
          else:
            n_p_nb = self.logsumexp(n_p_nb, p_b + p)
          
          # *NB* this would be a good place to include an LM score.
          next_beam[n_prefix] = (n_p_b, n_p_nb)

          # If s is repeated at the end we also update the unchanged
          # prefix. This is the merging case.
          if s == end_t:
            n_p_b, n_p_nb = next_beam[prefix]
            n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
            next_beam[prefix] = (n_p_b, n_p_nb)

        beam = sorted(next_beam.items(),
            key=lambda x : self.logsumexp(*x[1]),
            reverse=True)
        # Select best n
        beam = beam[:3]

     best = beam[0]
     return best[0], -1.0 * self.logsumexp(*best[1])


if __name__ == "__main__":
  np.random.seed(3)

  time = 50
  output_dim = 20

  probs = np.random.rand(time, output_dim)
  probs = probs / np.sum(probs, axis=1, keepdims=True)

  bms = beam_search()
  labels, score = bms.decode(probs)
  print("Score {:.3f}".format(score))
  print labels

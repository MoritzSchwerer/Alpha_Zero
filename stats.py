import pstats

p = pstats.Stats('profiler/gumbel_m1.prf')
p.sort_stats('tottime').print_stats(20)

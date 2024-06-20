import pstats

p = pstats.Stats('out3.prf')
p.sort_stats('tottime').print_stats(20)

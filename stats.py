import pstats

p = pstats.Stats('out2.prf')
p.sort_stats('cumulative').print_stats(20)

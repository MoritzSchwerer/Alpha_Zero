import pstats

p = pstats.Stats('profiler/sample_test1.prf')
p.sort_stats('tottime').print_stats(20)

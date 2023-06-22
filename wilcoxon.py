import scipy.stats as stats

# top 1 accuracy
group1 = [0.891, 0.834, 0.868]
group2 = [0.892, 0.842, 0.867]

#perform the Wilcoxon-Signed Rank Test
print(stats.wilcoxon(group1, group2))

# precision
group1 = [0.8922072984480892, 0.8776668219619587, 0.8842403421267464]
group2 = [0.8935503052142084, 0.8800587935312038, 0.8835827931419281]

#perform the Wilcoxon-Signed Rank Test
print(stats.wilcoxon(group1, group2))

# recall
group1 = [0.8941222221055568, 0.8309876009370071, 0.8736648522597027]
group2 = [0.8949555554388902, 0.8372128207651682, 0.8724780594074865]

#perform the Wilcoxon-Signed Rank Test
print(stats.wilcoxon(group1, group2))

# F1
group1 = [0.88636929173686, 0.8341807447667239, 0.8651385971153297]
group2 = [0.8875162952373201, 0.8407014188604034, 0.8633186681308642]

#perform the Wilcoxon-Signed Rank Test
print(stats.wilcoxon(group1, group2))

# top 5 accuracy
group1 = [0.989, 0.984, 0.994]
group2 = [0.99, 0.985, 0.994]

#perform the Wilcoxon-Signed Rank Test
print(stats.wilcoxon(group1, group2))


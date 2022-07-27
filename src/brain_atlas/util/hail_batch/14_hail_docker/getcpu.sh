while true; do 
	cgroup=$(awk -F: '$2 == "cpu,cpuacct" { print $3 }' /proc/self/cgroup)

	tstart=$(date +%s%N)
	cstart=$(cat /sys/fs/cgroup/cpu/cpuacct.usage)

	sleep 10

	tstop=$(date +%s%N)
	cstop=$(cat /sys/fs/cgroup/cpu/cpuacct.usage)


	read CPUUSAGE < <(bc -l <<< "define trunc(x) { auto s; s=scale; scale=0; x=x/1; scale=s; return x }; trunc(($cstop - $cstart) / ($tstop - $tstart) * 100)")



	echo -n 'CPU: '
	echo $CPUUSAGE | awk '{ printf("%03d", $1) }'

	MEM=$(awk '{ printf "%.2f\n", $1/1024/1024/1024; }'  /sys/fs/cgroup/memory/memory.max_usage_in_bytes)

	echo -n '% MMem: '
	echo -n $MEM
	echo "Gb"
        ps aufx
done

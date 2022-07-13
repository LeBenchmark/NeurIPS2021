
if [[ $# -eq 0 ]]; then
	echo "Usage: $0 <emission file list>"; exit;
fi

log_file=tmp.consumption.log
rm -f $log_file
while [[ $# -gt 0 ]]; do
	tail +2 $1 | cut -d',' -f4,5,6 | perl -pe '{s/,/ /g;}' >> $log_file
	shift
done

cat $log_file | awk 'BEGIN{time=0; kwh=0; co2=0;} {time=time+$1; kwh=kwh+$3; co2=co2+$2;} END{hours=int(time/3600); mins=int((time-hours*3600)/60); print "Time:",hours"h"mins"m.","KWh:",kwh,"CO2:",co2,"(KWh to CO2 factor:",kwh/co2"; Wh/minute",(kwh*1000)/(time/60)")"}'
rm -f tmp.consumption.log


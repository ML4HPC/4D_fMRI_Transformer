## ABCD_ADHD_TotalSymp_and_KSADS.csv 
This file contains the binary coded variables named as **ADHD_TotalSymp_and_KSADS** and **ADHD_TotalSymp_and_KSADS_and_CBCL**.  
  
* **ADHD_TotalSymp_and_KSADS**   
    * **ADHD_TotalSymp_and_KSADS** variable classify **ADHD** and **non-ADHD** by both criteria **ADHD total symptom scores ("adhd_total_symp")** and **parent report KSADS ("Attention.Deficit.Hyperactivity.Disorder.x")**.  
    * Subjects with *"adhd_total_symp" == 0  &  "Attention.Deficit.Hyperactivity.Disorder.x == 0"* are labeled as 0 (healthy).  
    * Subjects with *"adhd_total_symp" >= median(adhd_total_symp > 0)  &  "Attention.Deficit.Hyperactivity.Disorder.x == 1"* are labeled as 1 (case). **median(adhd_total_symp > 0) is 14**  
    * Subjects who have NA values in *ADHD total symptom scores* and *parent report KSADS* are omitted.  
    * Total subjects are *3,912* which contains *3,021* healthy subjects and *891* case subjects.  
  
* **ADHD_TotalSymp_and_KSADS_and_CBCL** 
    * **ADHD_TotalSymp_and_KSADS_and_CBCL** variable classify **ADHD** and **non-ADHD** by three criteria: **ADHD total symptom scores ("adhd_total_symp"), parent report KSADS ("Attention.Deficit.Hyperactivity.Disorder.x"), and CBCL ("ADHD.CBCL")**.  
    * Subjects with *"adhd_total_symp" == 0  &  "Attention.Deficit.Hyperactivity.Disorder.x == 0  &  "ADHD.CBCL == min(ADHD.CBCL)""* are labeled as 0 (healthy).  
    * Subjects with *"adhd_total_symp" >= median(adhd_total_symp > 0) & "Attention.Deficit.Hyperactivity.Disorder.x == 1"  &  "ADHD.CBCL > median(ADHD.CBCL > min(ADHD.CBCL))* are labeled as 1 (case). **median(adhd_total_symp > 0) is 14 and median(ADHD.CBCL > min(ADHD.CBCL)) is 56.**  
    * Subjects who have NA values in *ADHD total symptom scores* and *parent report KSADS* are omitted.  
    * Total subjects are *3,214* which contains *2,399* healthy subjects and *815* case subjects. 

## 1.abcd.bl.csv
This file contains **sub-scales of adhd symptom score** and **total score of adhd symptom ("adhd_total_symp")**. 
gcloud functions deploy export_hemodyn_to_gcs \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2GB \
  --timeout=1200 \


gcloud functions deploy export_lab_table_to_gcs \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2GB \
  --timeout=1200 \

gcloud projects add-iam-policy-binding dosewise-473716 \
  --member="serviceAccount:service-973837228669@dataflow-service-producer-prod.iam.gserviceaccount.com" \
  --role="roles/compute.networkUser"
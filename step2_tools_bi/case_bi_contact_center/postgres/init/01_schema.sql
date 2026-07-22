DROP TABLE IF EXISTS contact_center_training;

CREATE TABLE contact_center_training (
  week_start date NOT NULL,
  employee_id text NOT NULL,
  employee_name text NOT NULL,
  team text NOT NULL,
  supervisor text NOT NULL,
  calls_handled integer NOT NULL,
  sales_count integer NOT NULL,
  sales_revenue_rub numeric(14,2) NOT NULL,
  ai_errors_detected integer NOT NULL,
  supervisor_rejected_errors integer NOT NULL,
  recommendations_sent integer NOT NULL,
  recommendations_completed integer NOT NULL,
  training_minutes integer NOT NULL,
  qa_score numeric(6,2) NOT NULL,
  adherence_score numeric(8,5) NOT NULL,
  sales_per_call numeric(14,6) NOT NULL,
  rejection_rate numeric(14,6) NOT NULL,
  recommendation_uptake numeric(14,6) NOT NULL
);

CREATE INDEX idx_contact_center_training_week ON contact_center_training (week_start);
CREATE INDEX idx_contact_center_training_employee ON contact_center_training (employee_id);

#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.fill(0);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.fill(0);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acXsig_pred_celeration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 5 + 2;
  lambda_ = 3 - n_aug_;

  // set weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < 2 * n_aug_ + 1; i++)
  { //2n+1 weights
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }
  pre_time_ = 0;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_)
  {
    return;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_)
  {
    return;
  }

  double delta_t = (double)(meas_package.timestamp_ - pre_time_) / 1e6;
  pre_time_ = meas_package.timestamp_;
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);
      x_ << px, py, 0, 0, 0;
    }
    else
    { // Radar
      float ro = meas_package.raw_measurements_(0);
      float theta = meas_package.raw_measurements_(1);
      float ro_dot = meas_package.raw_measurements_(2);
      float px = ro * cos(theta);
      float py = ro * sin(theta);

      x_ << px, py, ro_dot, theta, 0;
    }

    P_ = MatrixXd::Identity(n_x_, n_x_);
    P_(2, 2) = std_a_ * std_a_;
    P_(3, 3) = std_yawdd_ * std_yawdd_;

    is_initialized_ = true;
    return;
  }

  Prediction(delta_t);
  if (meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else
  { // Radar
    UpdateRadar(meas_package);
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  AugmentedSigmaPoints();
  SigmaPointPrediction(delta_t);
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  VectorXd z_pred;
  MatrixXd S;
  MatrixXd Zsig;
  PredictLaserMeasurement(z_pred, S, Zsig);
  UpdateState(Zsig, z_pred, S, meas_package.raw_measurements_, 2);

  //VectorXd temp = meas_package.raw_measurements_ - z_pred;
  //NIS_lidar_ = temp.transpose() * S.inverse() * temp;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  VectorXd z_pred;
	MatrixXd S;
	MatrixXd Zsig;
	PredictRadarMeasurement(z_pred,S,Zsig);
	UpdateState(Zsig,z_pred,S,meas_package.raw_measurements_,3);

	// VectorXd temp = meas_package.raw_measurements_ - z_pred;
	// NIS_radar_ = temp.transpose() * S.inverse() * temp;
}

void UKF::AugmentedSigmaPoints(void)
{
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0);

  //create sigma point matrix
  Xsig_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_.fill(0);

  //create augmented mean state
  x_aug.block(0, 0, n_x_, 1) = x_;
  x_aug.block(n_x_, 0, n_aug_ - n_x_, 1) << 0, 0;
  //create augmented covariance matrix
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug.block(n_x_, n_x_, n_aug_ - n_x_, n_aug_ - n_x_) << std_a_ * std_a_, 0,
      0, std_yawdd_ * std_yawdd_;
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  //create augmented sigma points
  //set first column of sigma point matrix
  Xsig_.col(0) = x_aug;

  //set remaining sigma points
  for (int i = 0; i < n_aug_; i++)
  {
    Xsig_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }
}

void UKF::SigmaPointPrediction(double delta_t)
{
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  for (int i = 0; i < Xsig_pred.cols(); i++)
  {
    double px = Xsig_(0, i), py = Xsig_(1, i), v = Xsig_(2, i), psi = Xsig_(3, i), psi_dot = Xsig_(4, i);
    double va = Xsig_(5, i), vpsi = Xsig_(6, i);
    double delta_t2 = delta_t * delta_t;
    double x0, x1;
    if (fabs(psi_dot) > 1e-6)
    { // not zero
      x0 = px + v / psi_dot * (sin(psi + psi_dot * delta_t) - sin(psi)) + 0.5 * delta_t2 * cos(psi) * va;
      x1 = py + v / psi_dot * (-cos(psi + psi_dot * delta_t) + cos(psi)) + 0.5 * delta_t2 * sin(psi) * va;
    }
    else
    {
      x0 = px + v * cos(psi) * delta_t + 0.5 * delta_t2 * cos(psi) * va;
      x1 = py + v * sin(psi) * delta_t + 0.5 * delta_t2 * sin(psi) * va;
    }
    double x2 = v + delta_t * va;
    double x3 = psi + psi_dot * delta_t + 0.5 * delta_t2 * vpsi;
    double x4 = psi_dot + delta_t * vpsi;

    Xsig_pred.col(i) << x0, x1, x2, x3, x4;
  }

  //write result
  Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(void)
{
  int n_sig = Xsig_pred_.cols();
  x_.fill(0);
  for (int i = 0; i < n_sig; i++)
  {
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }
  P_.fill(0);
  for (int i = 0; i < n_sig; i++)
  {
    VectorXd temp = Xsig_pred_.col(i) - x_;
    P_ = P_ + (weights_(i) * temp) * temp.transpose();
  }
}

void UKF::UpdateState(Eigen::MatrixXd &Zsig, Eigen::VectorXd &z_pred, Eigen::MatrixXd &S, Eigen::VectorXd &z, int n_z)
{
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  for (int i = 0; i < Xsig_pred_.cols(); i++)
  {
    VectorXd temp = Zsig.col(i) - z_pred;
    if (n_z == 3)
    { // Radar
      //angle normalization
      normalizeAngle(temp(1));
    }
    Tc = Tc + weights_(i) * (Xsig_pred_.col(i) - x_) * temp.transpose();
  }

  MatrixXd Kg = Tc * S.inverse();

  x_ = x_ + Kg * (z - z_pred);

  P_ = P_ - Kg * S * Kg.transpose();
}

void UKF::PredictRadarMeasurement(Eigen::VectorXd &z_out, Eigen::MatrixXd &S_out, Eigen::MatrixXd &Zsig)
{
  int n_z = 3;
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);

  for (int i = 0; i < Xsig_pred_.cols(); i++)
  {
    double px = Xsig_pred_(0, i), py = Xsig_pred_(1, i), v = Xsig_pred_(2, i), psi = Xsig_pred_(3, i); //,psi_dot=Xsig_pred_(4,i);
    double x0 = sqrt(px * px + py * py);
    double x1 = atan2(py, px);
    double x2 = (px * cos(psi) * v + py * sin(psi) * v) / x0;
    Zsig.col(i) << x0, x1, x2;
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;
  for (int i = 0; i < Xsig_pred_.cols(); i++)
  {
    VectorXd temp = Zsig.col(i) - z_pred;
    //angle normalization
    normalizeAngle(temp(1));
    S = S + (weights_(i) * temp) * temp.transpose();
  }
  S = S + R;
  z_out = z_pred;
  S_out = S;
}

void UKF::PredictLaserMeasurement(Eigen::VectorXd &z_out, Eigen::MatrixXd &S_out, Eigen::MatrixXd &Zsig)
{
  int n_z = 2;
  Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0);

  for (int i = 0; i < Xsig_pred_.cols(); i++)
  {
    double px = Xsig_pred_(0, i), py = Xsig_pred_(1, i);
    double x0 = px;
    double x1 = py;
    Zsig.col(i) << x0, x1;
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;

  for (int i = 0; i < Xsig_pred_.cols(); i++)
  {
    VectorXd temp = Zsig.col(i) - z_pred;
    S = S + (weights_(i) * temp) * temp.transpose();
  }
  S = S + R;
  z_out = z_pred;
  S_out = S;
}

void UKF::normalizeAngle(double &angle)
{
  while (angle > M_PI)
  {
    angle -= 2. * M_PI;
  }
  while (angle < -M_PI)
  {
    angle += 2. * M_PI;
  }
}
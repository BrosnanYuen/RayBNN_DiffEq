//cargo run --example  Linear_Matrix_ODE --release

use arrayfire;
use RayBNN_DiffEq;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

fn main() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);


	//Create A matrix from random normal numbers
	let A_dims = arrayfire::Dim4::new(&[1000,1000,1,1]);
	let A = arrayfire::randn::<f64>(A_dims)/100.0f64;

	// Set the Linear Matrix Differentail Equation
	// dy/dt = A*y
	let diffeq = |t: &arrayfire::Array<f64>, y: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		arrayfire::matmul(&A, y, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE)
	};

	//Start at t=0 and end at t=50
	//Step size of 0.001
	//Relative error of 1E-9
	//Absolute error of 1E-9
	//Error Type compute the individual error of every element in y
	let options: RayBNN_DiffEq::ODE::ODE45::ODE45_Options<f64> = RayBNN_DiffEq::ODE::ODE45::ODE45_Options {
		tstart: 0.0f64,
		tend: 50.0f64,
		tstep: 0.001f64,
		rtol: 1.0E-9f64,
		atol: 1.0E-9f64,
		error_select: RayBNN_DiffEq::ODE::ODE45::error_type::INDIVIDUAL_ERROR
	};

	let t_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut t = arrayfire::constant::<f64>(0.0,t_dims);

	let y0_dims = arrayfire::Dim4::new(&[1000,1,1,1]);
	let mut y = arrayfire::constant::<f64>(0.0,y0_dims);
	let mut dydt = arrayfire::constant::<f64>(0.0,y0_dims);

	//Initial Point of Differential Equation
	let y0 = arrayfire::randn::<f64>(y0_dims)/100.0f64;
	

	println!("Running");

	arrayfire::sync(DEVICE);
	let starttime = std::time::Instant::now();

	//Run Solver
	RayBNN_DiffEq::ODE::ODE45::linear_ode_solve(
		&y0
		,diffeq
		,&options
		,&mut t
		,&mut y
		,&mut dydt
	);

	arrayfire::sync(DEVICE);

	let elapsedtime = starttime.elapsed();
	
	arrayfire::sync(DEVICE);

	//let lasty = arrayfire::col(&y, y.dims()[1] as i64);
	//arrayfire::print_gen("lasty".to_string(), &lasty,Some(6));
	//arrayfire::print_gen("t".to_string(), &t,Some(6));

	println!("Computed {} Steps In: {:.6?}", y.dims()[1],elapsedtime);


}

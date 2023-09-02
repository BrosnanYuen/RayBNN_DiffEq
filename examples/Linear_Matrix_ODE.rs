//cargo run --example  Linear_Matrix_ODE --release

use arrayfire;
use RayBNN_DiffEq;

//Select CUDA and GPU Device 0
const BACK_END: arrayfire::Backend = arrayfire::Backend::CUDA;
const DEVICE: i32 = 0;

fn main() {

	arrayfire::set_backend(BACK_END);
	arrayfire::set_device(DEVICE);


	//Create A matrix
	let A_vec:Vec<f64> = vec![1.0, 1.2, 1.1,    0.8, -1.0, 0.0,    0.0, 0.0, -1.2];
	let mut A = arrayfire::Array::new(&A_vec, arrayfire::Dim4::new(&[3, 3, 1, 1]));
	arrayfire::print_gen("A".to_string(), &A,Some(6));

	//A
    //1.000000     0.800000     0.000000 
    //1.200000    -1.000000     0.000000 
    //1.100000     0.000000    -1.200000 

	// Set the Linear Matrix Differentail Equation
	// dy1/dt = 1.0y1 + 0.8y2  + 0.0y3
	// dy2/dt = 1.2y1 + -1.0y2 + 0.0y3
	// dy3/dt = 1.1y1 + 0.0y2  + -1.2y3 
	let diffeq = |t: &arrayfire::Array<f64>, y: &arrayfire::Array<f64>| -> arrayfire::Array<f64> {
		arrayfire::matmul(&A, y, arrayfire::MatProp::NONE, arrayfire::MatProp::NONE)
	};

	//Start at t=0 and end at t=10
	//Step size of 0.001
	//Relative error of 1E-9
	//Absolute error of 1E-9
	//Error Type compute the total error of every element in y
	let options: RayBNN_DiffEq::ODE::ODE45::ODE45_Options<f64> = RayBNN_DiffEq::ODE::ODE45::ODE45_Options {
		tstart: 0.0f64,
		tend: 10.0f64,
		tstep: 0.001f64,
		rtol: 1.0E-9f64,
		atol: 1.0E-9f64,
		error_select: RayBNN_DiffEq::ODE::ODE45::error_type::TOTAL_ERROR
	};

	let t_dims = arrayfire::Dim4::new(&[1,1,1,1]);
	let mut t = arrayfire::constant::<f64>(0.0,t_dims);

	let y0_dims = arrayfire::Dim4::new(&[3,1,1,1]);
	let mut y = arrayfire::constant::<f64>(0.0,y0_dims);
	let mut dydt = arrayfire::constant::<f64>(0.0,y0_dims);

	//Initial Point of Differential Equation
	//Set y1(0) = 0.1
	//Set y2(0) = 0.2
	//Set y3(0) = -0.3
	let y0_vec:Vec<f64> = vec![0.1, 0.2, -0.3];
	let y0 = arrayfire::Array::new(&y0_vec, y0_dims);
	

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

	arrayfire::print_gen("y".to_string(), &y,Some(6));
	arrayfire::print_gen("t".to_string(), &t,Some(6));

	println!("Computed {} Steps In: {:.6?}", y.dims()[1],elapsedtime);


}

use super::Graph;
use super::Output;
use super::Result;
use super::Status;
use std::ffi::CString;
use std::ffi::NulError;
use std::mem;
use std::os::raw::c_int;
use std::ptr;
use std::result;
use std::slice;
use tensorflow_sys_runtime as tf;

// This exists purely to ensure TF_AbortWhile gets called properly, even on panic.
#[derive(Debug)]
struct CWhileParams {
    inner: tf::TF_WhileParams,
    finished: bool,
}

impl Drop for CWhileParams {
    fn drop(&mut self) {
        if !self.finished {
            unsafe {
                tf::TF_AbortWhile(&self.inner);
            }
        }
    }
}

/// A WhileBuilder is used to build a while loop.
#[derive(Debug)]
pub struct WhileBuilder<'a> {
    graph: &'a mut Graph,
    inner: CWhileParams,
    name: Option<CString>,
    #[allow(dead_code)]
    c_inputs: Vec<tf::TF_Output>, // must live until TF_FinishWhile
}

impl<'a> WhileBuilder<'a> {
    /// Creates a WhileBuilder for creating a while loop.
    ///
    /// The loop is not actually added to the graph until `finish` is called.
    ///
    /// # Arguments
    ///
    /// * `graph` - graph to create the while loop in
    /// * `cond` - takes the condition subgraph and loop variables and returns a
    ///   scalar boolean
    /// * `body` - takes the body subgraph and loop variables and returns the new
    ///   values of the loop variables
    /// * `inputs` - outputs that already exist in `graph` used as initial values
    ///   for the loop variables
    ///
    /// # Returns
    ///
    /// A unfinished WhileBuilder.
    pub fn new<
        CF: Fn(&mut Graph, &[Output]) -> Result<Output>,
        BF: Fn(&mut Graph, &[Output]) -> Result<Vec<Output>>,
    >(
        graph: &'a mut Graph,
        cond: CF,
        body: BF,
        inputs: &[Output],
    ) -> Result<Self> {
        let mut status = Status::new();
        let c_inputs: Vec<_> = inputs.iter().map(Output::to_c).collect();
        let mut inner = CWhileParams {
            inner: unsafe {
                tf::TF_NewWhile(
                    graph.inner(),
                    c_inputs.as_ptr() as *mut _,
                    c_inputs.len() as c_int,
                    status.inner(),
                )
            },
            finished: false,
        };
        if let Err(e) = status.into_result() {
            inner.finished = true; // used by Drop impl
            return Err(e);
        }

        // Fill in the condition graph
        let mut cond_graph = unsafe { Graph::from_c(inner.inner.cond_graph as *mut _) };
        let cond_inputs: Vec<_> = unsafe {
            slice::from_raw_parts(inner.inner.cond_inputs, inputs.len())
                .iter()
                .map(|out| Output::from_c(graph, out))
                .collect()
        };
        let cond_out = cond(&mut cond_graph, &cond_inputs)?;
        inner.inner.cond_output = cond_out.to_c();

        // Fill in the body graph
        let mut body_graph = unsafe { Graph::from_c(inner.inner.body_graph as *mut _) };
        let body_inputs: Vec<_> = unsafe {
            slice::from_raw_parts(inner.inner.body_inputs, inputs.len())
                .iter()
                .map(|out| Output::from_c(graph, out))
                .collect()
        };
        let body_out = body(&mut body_graph, &body_inputs)?;
        if body_out.len() != inputs.len() {
            return Err(invalid_arg!(
                "Expected {} outputs, but got {}",
                inputs.len(),
                body_out.len()
            ));
        }
        let c_body_out =
            unsafe { slice::from_raw_parts_mut(inner.inner.body_outputs as *mut _, inputs.len()) };
        for i in 0..inputs.len() {
            c_body_out[i] = body_out[i].to_c();
        }

        Ok(WhileBuilder {
            graph,
            inner,
            name: None,
            c_inputs,
        })
    }

    /// Sets a unique name for this while loop. This is used as a prefix
    /// for created operations. If not set, a unique prefix will be generated.
    pub fn name(mut self, name: &str) -> result::Result<Self, NulError> {
        self.name = Some(CString::new(name)?);
        Ok(self)
    }

    /// Builds the while loop and returns the output tensors of the while loop.
    pub fn finish(mut self) -> Result<Vec<Output>> {
        let status = Status::new();
        let mut c_outputs: Vec<tf::TF_Output> =
            Vec::with_capacity(self.inner.inner.ninputs as usize);

        let mut name = None;
        mem::swap(&mut self.name, &mut name);
        let name = match name {
            None => {
                // Include /Merge because while_loop_{} doesn't describe an
                // operation on its own.
                let while_loop_index = self.graph.generate_operation_name("while_loop_{}/Merge")?;
                CString::new(format!("while_loop_{}", while_loop_index))?
            }
            Some(name) => name,
        };
        self.inner.inner.name = name.as_ptr();

        unsafe {
            c_outputs.set_len(self.inner.inner.ninputs as usize);
            for c_output in &mut c_outputs {
                // For some reason, these have to be initialized to {null, -1},
                // even though they're output parameters.
                c_output.oper = ptr::null_mut();
                c_output.index = -1;
            }
            self.inner.finished = true; // used by Drop impl
            tf::TF_FinishWhile(&self.inner.inner, status.inner, c_outputs.as_mut_ptr());
        }
        status.into_result()?;
        Ok(c_outputs
            .iter()
            .map(|out| Output::from_c(self.graph, out))
            .collect())
    }
}

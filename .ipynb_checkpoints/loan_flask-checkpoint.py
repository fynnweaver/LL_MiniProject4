# import Flask and jsonify
from flask import Flask, jsonify, request
# import Resource, Api and reqparser
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle

#standard flask init
app = Flask(__name__)
api = Api(app)

#our model needs access to our custom functions
from custom_functions import impute, dropID, numFeat, catFeat, log_transform, cat_transform

#load model
model = pickle.load(open('loan_model.sav', 'rb'))


#now we want an endpoint to communicate with ML, this time a post request
class loan_predict(Resource):
    def post(self):
        #transform data posted into a dataframe
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        
        #getting predictions from the model using our pipeline
        res = model.predict_proba(df)
        
        #translate res to a list and return (can't return np arrays)
        return res.tolist()
    
    def get(self):
         #create request parser
        parser = reqparse.RequestParser()
        num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History']
        cat_cols = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Property_Area']
        
        #create arguments
        for col in num_cols:
            parser.add_argument(col, type = float)
            
        for col in cat_cols:
            parser.add_argument(col, type = str)
        
        cols = num_cols + cat_cols
        input_values = {'Loan_ID': 0}
        
        #assign input arguments into objects
        for col in cols:
            if request.args.get(col) is not None:
                input_values[col] = parser.parse_args().get(col)
            else:
                input_values[col] = np.nan
        
        df = pd.DataFrame(input_values.values(), index=input_values.keys()).T
        res = model.predict_proba(df)
        
        return res.tolist()
        
        
    
    
#Now to assing an endpoint to our API
api.add_resource(loan_predict, '/loan_prediction')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5555)